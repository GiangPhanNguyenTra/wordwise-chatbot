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
    
    uk_detail: Optional[PhoneticDetail] = None
    us_detail: Optional[PhoneticDetail] = None
    
    phonetics_list = raw_data.get('phonetics', [])
    other_candidates: List[PhoneticDetail] = []

    if isinstance(phonetics_list, list):
        for item in phonetics_list:
            if not isinstance(item, dict):
                continue
            
            text = item.get('text', '')
            audio_url = item.get('audio', '')

            if not text and not audio_url:
                continue
            
            detail = PhoneticDetail(text=text, audio=audio_url)
            
            audio_lower = audio_url.lower()
            if ('uk.mp3' in audio_lower or '-uk' in audio_lower) and not uk_detail:
                uk_detail = detail
            elif ('us.mp3' in audio_lower or '-us' in audio_lower) and not us_detail:
                us_detail = detail
            else:
                other_candidates.append(detail)
        
        if not uk_detail and other_candidates:
            for cand in other_candidates:
                if cand != us_detail:
                    uk_detail = cand
                    break
            if not uk_detail and other_candidates:
                uk_detail = other_candidates[0]
                
        if not us_detail and other_candidates:
            for cand in other_candidates:
                if cand != uk_detail:
                    us_detail = cand
                    break
            if not us_detail and other_candidates:
                us_detail = other_candidates[0]

        if us_detail and not uk_detail:
            uk_detail = us_detail
        elif uk_detail and not us_detail:
            us_detail = uk_detail

    final_phonetics = Phonetics(uk=uk_detail, us=us_detail)
    
    phonetics_dump = final_phonetics.model_dump(exclude_none=True)
    if phonetics_dump:
        raw_data['phonetics'] = phonetics_dump
    else:
        raw_data.pop('phonetics', None)

    meanings = raw_data.get('meanings', [])
    if isinstance(meanings, list) and meanings:
        primary_meaning = meanings[0]
        definitions = primary_meaning.get('definitions', [])
        if definitions and isinstance(definitions[0], dict):
            raw_data['main_definition'] = definitions[0].get('definition', '')
            raw_data['main_example'] = definitions[0].get('example', '')
        raw_data['main_partOfSpeech'] = primary_meaning.get('partOfSpeech', '')
    
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