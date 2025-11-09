import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.models.word import EnrichedWord, TranslatedPhrase, Phonetics, PhoneticDetail
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.dictionary_tool import fetch_dictionary_data
from app.utils.helpers import load_prompt

def _filter_and_deduplicate_phrases(
    word: str, 
    phrases: List[TranslatedPhrase], 
    require_root_word: bool
) -> List[TranslatedPhrase]:
    seen = set()
    unique_phrases = []
    word_lower = word.lower()

    for phrase_pair in phrases:
        if not (isinstance(phrase_pair, TranslatedPhrase) and phrase_pair.en):
            continue
            
        phrase_en = phrase_pair.en.lower().strip()
        normalized_phrase = phrase_en.removeprefix("to ").removeprefix("to be ")
        
        # 1. QUY TẮC PHẢI LÀ CỤM TỪ (> 1 từ)
        if len(phrase_en.split()) <= 1:
            continue
            
        # 2. QUY TẮC PHẢI CHỨA TỪ GỐC (nếu được yêu cầu)
        if require_root_word and word_lower not in phrase_en:
            continue

        # 3. LOẠI BỎ TRÙNG LẶP
        if normalized_phrase not in seen:
            seen.add(normalized_phrase)
            unique_phrases.append(phrase_pair)
            
    return unique_phrases

def _post_process_enriched_data(data: EnrichedWord) -> EnrichedWord:
    # 1. ÁP DỤNG LỌC NGHIÊM NGẶT CHO IDIOMS_COLLOCATIONS (PHẢI CHỨA TỪ GỐC)
    if data.idioms_collocations:
        data.idioms_collocations = _filter_and_deduplicate_phrases(
            data.word, data.idioms_collocations, require_root_word=True
        )

    # 2. ÁP DỤNG LỌC CHO PHRASAL_VERBS (Không cần chứa từ gốc, nhưng > 1 từ)
    if data.phrasal_verbs:
        data.phrasal_verbs = _filter_and_deduplicate_phrases(
            data.word, data.phrasal_verbs, require_root_word=False
        )

    # 3. Lớp bảo vệ chống Phrasal Verbs ảo
    if data.partOfSpeech != 'verb' and data.phrasal_verbs:
        data.phrasal_verbs = []
        
    # 4. Lớp bảo vệ chống trùng lặp chéo
    if data.idioms_collocations and data.phrasal_verbs:
        phrasal_verb_texts = {pv.en.lower().strip() for pv in data.phrasal_verbs}
        data.idioms_collocations = [
            ic for ic in data.idioms_collocations 
            if ic.en.lower().strip() not in phrasal_verb_texts
        ]
        
    return data


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

async def enrich_word_data(word: str, context: Optional[str] = None) -> EnrichedWord:
    raw_data = await fetch_dictionary_data(word)
    
    processed_raw_data = _pre_process_dictionary_data(raw_data)
    
    raw_data_str = str(processed_raw_data) if processed_raw_data else "No dictionary data found."
    
    enrich_prompt_cfg = load_prompt("enrichment")
    
    prompt_kwargs = {
        "word": word,
        "raw_data": raw_data_str,
        "context": context if context else ""
    }
    
    enriched_data = await structured_llm_call(
        enrich_prompt_cfg["template"],
        EnrichedWord,
        **prompt_kwargs
    )
    
    return _post_process_enriched_data(enriched_data)

class BulkEnrichedWordsResponse(BaseModel):
    results: List[EnrichedWord]

async def batch_enrich_words_data(words: List[str]) -> List[EnrichedWord]:
    if not words:
        return []
    
    prompt_config = load_prompt("batch_enrich_words")
    
    # Chuyển danh sách từ thành một chuỗi JSON để đưa vào prompt
    words_json_string = json.dumps(words)
    
    try:
        response_model = await structured_llm_call(
            prompt_config["template"],
            BulkEnrichedWordsResponse,
            words_json_string=words_json_string
        )
        return response_model.results
    except Exception as e:
        print(f"Failed to batch enrich words: {e}")
        return []