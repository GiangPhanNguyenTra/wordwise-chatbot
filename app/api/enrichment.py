import asyncio
from fastapi import APIRouter, HTTPException
from app.models.word import EnrichedWord
from app.models.enrichment import BulkEnrichRequest, BulkEnrichResponse
from app.core.tools.enrichment_tool import enrich_word_data

router = APIRouter()

@router.get(
    "/{word}", 
    response_model=EnrichedWord,
    summary="Enrich a single word"
)
async def enrich_single_word_endpoint(word: str):
    try:
        enriched_result = await enrich_word_data(word)
        return enriched_result
    except Exception as e:
        print(f"Error enriching single word '{word}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enrich word: {word}")

@router.post(
    "/bulk", 
    response_model=BulkEnrichResponse,
    summary="Enrich a list of words in parallel"
)
async def enrich_bulk_words_endpoint(request: BulkEnrichRequest):
    if not request.words:
        return BulkEnrichResponse(results=[])

    try:
        tasks = [enrich_word_data(word) for word in request.words]
        enriched_results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results = [res for res in enriched_results if not isinstance(res, Exception)]
        
        return BulkEnrichResponse(results=successful_results)
    except Exception as e:
        print(f"Error in bulk enrichment: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during bulk processing.")