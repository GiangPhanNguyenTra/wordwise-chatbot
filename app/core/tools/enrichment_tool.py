# app/core/tools/enrichment_tool.py
from typing import List
from app.models.word import EnrichedWord, TranslatedPhrase
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.dictionary_tool import fetch_dictionary_data
from app.utils.helpers import load_prompt

def _normalize_and_deduplicate_phrases(phrases: List[TranslatedPhrase]) -> List[TranslatedPhrase]:
    seen = set()
    unique_phrases = []
    for phrase_pair in phrases:
        normalized_phrase = phrase_pair.en.lower().strip().removeprefix("to ").removeprefix("to be ")
        if normalized_phrase not in seen:
            seen.add(normalized_phrase)
            unique_phrases.append(phrase_pair)
    return unique_phrases

async def enrich_word_data(word: str) -> EnrichedWord:
    raw_data = await fetch_dictionary_data(word)
    raw_data_str = str(raw_data) if raw_data else "No dictionary data found."
    
    enrich_prompt_cfg = load_prompt("enrichment")
    enriched_data = await structured_llm_call(
        enrich_prompt_cfg["template"],
        EnrichedWord,
        word=word,
        raw_data=raw_data_str
    )
    
    if enriched_data.idioms:
        enriched_data.idioms = _normalize_and_deduplicate_phrases(enriched_data.idioms)
    if enriched_data.collocations:
        enriched_data.collocations = _normalize_and_deduplicate_phrases(enriched_data.collocations)
    if enriched_data.phrasal_verbs:
        enriched_data.phrasal_verbs = _normalize_and_deduplicate_phrases(enriched_data.phrasal_verbs)
    
    return enriched_data