# app/core/tools/enrichment_tool.py
from typing import List, Dict, Any, Optional
from app.models.word import EnrichedWord, TranslatedPhrase, Phonetics, PhoneticDetail
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.dictionary_tool import fetch_dictionary_data
from app.utils.helpers import load_prompt

def _normalize_and_deduplicate_phrases(phrases: List[TranslatedPhrase]) -> List[TranslatedPhrase]:
    seen = set()
    unique_phrases = []
    for phrase_pair in phrases:
        if not (isinstance(phrase_pair, TranslatedPhrase) and phrase_pair.en):
            continue
        normalized_phrase = phrase_pair.en.lower().strip().removeprefix("to ").removeprefix("to be ")
        if normalized_phrase not in seen:
            seen.add(normalized_phrase)
            unique_phrases.append(phrase_pair)
    return unique_phrases

def _pre_process_dictionary_data(raw_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw_data:
        return None

    print("--- Pre-processing raw dictionary data ---")
    
    # Xử lý phonetics
    phonetics_list = raw_data.get('phonetics', [])
    if isinstance(phonetics_list, list):
        new_phonetics = Phonetics()
        uk_found, us_found = False, False
        for item in phonetics_list:
            if not isinstance(item, dict): continue
            audio_url = item.get('audio', '')
            if 'uk.mp3' in audio_url and not uk_found:
                new_phonetics.uk = PhoneticDetail(text=item.get('text'), audio=audio_url)
                uk_found = True
            elif 'us.mp3' in audio_url and not us_found:
                new_phonetics.us = PhoneticDetail(text=item.get('text'), audio=audio_url)
                us_found = True
        
        # Fallback logic
        if not uk_found and len(phonetics_list) > 0 and isinstance(phonetics_list[0], dict):
             new_phonetics.uk = PhoneticDetail(text=phonetics_list[0].get('text'), audio=phonetics_list[0].get('audio'))
        if not us_found and len(phonetics_list) > 1 and isinstance(phonetics_list[1], dict):
             new_phonetics.us = PhoneticDetail(text=phonetics_list[1].get('text'), audio=phonetics_list[1].get('audio'))
             
        raw_data['phonetics'] = new_phonetics.model_dump(exclude_none=True)

    # Chắt lọc definitions và examples
    meanings = raw_data.get('meanings', [])
    if isinstance(meanings, list) and meanings:
        primary_meaning = meanings[0]
        definitions = primary_meaning.get('definitions', [])
        if definitions and isinstance(definitions[0], dict):
            raw_data['main_definition'] = definitions[0].get('definition', '')
            raw_data['main_example'] = definitions[0].get('example', '')
        raw_data['main_partOfSpeech'] = primary_meaning.get('partOfSpeech', '')
    
    # Dọn dẹp các trường không cần thiết để giảm context cho LLM
    raw_data.pop('meanings', None)
    raw_data.pop('license', None)
    
    return raw_data

async def enrich_word_data(word: str) -> EnrichedWord:
    raw_data = await fetch_dictionary_data(word)
    
    processed_raw_data = _pre_process_dictionary_data(raw_data)
    
    raw_data_str = str(processed_raw_data) if processed_raw_data else "No dictionary data found."
    
    enrich_prompt_cfg = load_prompt("enrichment")
    enriched_data = await structured_llm_call(
        enrich_prompt_cfg["template"],
        EnrichedWord,
        word=word,
        raw_data=raw_data_str
    )
    
    if enriched_data.idioms_collocations:
        enriched_data.idioms_collocations = _normalize_and_deduplicate_phrases(enriched_data.idioms_collocations)
    if enriched_data.phrasal_verbs:
        enriched_data.phrasal_verbs = _normalize_and_deduplicate_phrases(enriched_data.phrasal_verbs)
    if enriched_data.partOfSpeech != 'verb' and enriched_data.phrasal_verbs:
        enriched_data.phrasal_verbs = []
        
    return enriched_data