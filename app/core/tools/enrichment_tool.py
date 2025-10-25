from typing import List
from app.models.word import EnrichedWord, IdiomPair
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.dictionary_tool import fetch_dictionary_data
from app.utils.helpers import load_prompt

def _normalize_and_deduplicate_idioms(idioms: List[IdiomPair]) -> List[IdiomPair]:
    seen = set()
    unique_idioms = []
    for idiom_pair in idioms:
        normalized_idiom = idiom_pair.en.lower().strip().removeprefix("to ").removeprefix("to be ")
        if normalized_idiom not in seen:
            seen.add(normalized_idiom)
            unique_idioms.append(idiom_pair)
    return unique_idioms

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
        enriched_data.idioms = _normalize_and_deduplicate_idioms(enriched_data.idioms)
    
    return enriched_data