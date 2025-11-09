from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from app.services.core_service import core_service
from app.utils.helpers import load_prompt
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.enrichment_tool import batch_enrich_words_data
from app.models.word import EnrichedWord

router = APIRouter()

class RecommendRequest(BaseModel):
    jwt: str = Field(..., description="The JWT token for user authentication.")

class WordRecommendation(BaseModel):
    word: str
    collection: str

class WordRecommendationList(BaseModel):
    recommendations: List[WordRecommendation]

class EnrichedRecommendedWord(EnrichedWord):
    recommended_collection: str

class RecommendResponse(BaseModel):
    results: List[EnrichedRecommendedWord]

@router.post(
    "/recommendations",
    response_model=RecommendResponse,
    summary="Generate and enrich vocabulary recommendations"
)
async def get_recommendations(request: RecommendRequest):
    collection_names = await core_service.get_collection_names(request.jwt)
    if not collection_names:
        raise HTTPException(
            status_code=404, 
            detail="Could not find any collections for the user or failed to fetch them. Please create a collection first."
        )

    try:
        topics_str = ", ".join(collection_names)
        prompt_config = load_prompt("recommend_words")
        llm_response = await structured_llm_call(
            prompt_config["template"],
            WordRecommendationList,
            topics=topics_str
        )
        recommendations = llm_response.recommendations
    except Exception as e:
        print(f"Error getting recommendations from LLM: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations from AI.")
    
    if not recommendations:
        return RecommendResponse(results=[])

    words_to_enrich = [rec.word for rec in recommendations]
    enriched_results = await batch_enrich_words_data(words_to_enrich)

    if not enriched_results:
        raise HTTPException(status_code=500, detail="Failed to enrich the recommended words.")

    collection_map = {rec.word: rec.collection for rec in recommendations}
    final_results = []
    
    for enriched_word in enriched_results:
        collection_name = collection_map.get(enriched_word.word)
        if collection_name:
            final_results.append(
                EnrichedRecommendedWord(
                    **enriched_word.model_dump(),
                    recommended_collection=collection_name
                )
            )

    return RecommendResponse(results=final_results)