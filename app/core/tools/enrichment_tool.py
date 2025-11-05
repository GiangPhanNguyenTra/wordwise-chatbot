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

    if isinstance(phonetics_list, list):
        # Ưu tiên 1: Tìm chính xác phiên âm UK và US dựa vào audio URL
        for item in phonetics_list:
            if not (isinstance(item, dict) and item.get('text') and item.get('audio')):
                continue 
            
            audio_url = item['audio']
            text = item['text']

            if 'uk.mp3' in audio_url and not uk_detail:
                uk_detail = PhoneticDetail(text=text, audio=audio_url)
            elif 'us.mp3' in audio_url and not us_detail:
                us_detail = PhoneticDetail(text=text, audio=audio_url)
        
        # Ưu tiên 2 (Dự phòng): Nếu vẫn thiếu, lấy bất kỳ phiên âm nào có sẵn để lấp chỗ trống
        if not uk_detail or not us_detail:
            used_texts = {p.text for p in [uk_detail, us_detail] if p}
            available_phonetics = [
                p for p in phonetics_list 
                if isinstance(p, dict) and p.get('text') and p.get('text') not in used_texts
            ]
            
            if not uk_detail and available_phonetics:
                fallback_item = available_phonetics.pop(0)
                uk_detail = PhoneticDetail(text=fallback_item.get('text'), audio=fallback_item.get('audio', ''))

            if not us_detail and available_phonetics:
                fallback_item = available_phonetics.pop(0)
                us_detail = PhoneticDetail(text=fallback_item.get('text'), audio=fallback_item.get('audio', ''))


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