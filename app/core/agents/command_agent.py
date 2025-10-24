# app/core/agents/command_agent.py

import uuid
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.models.chat import AgentState
from app.models.word import EnrichedWord
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.dictionary_tool import fetch_dictionary_data
from app.utils.helpers import load_prompt
# SỬA Ở ĐÂY: Import class MongoService
from app.services.mongo_service import MongoService

class CommandExtraction(BaseModel):
    word: Optional[str] = None
    collection: Optional[str] = None

async def _enrich_word_data(word: str) -> EnrichedWord:
    """Helper function to perform the full enrichment flow."""
    raw_data = await fetch_dictionary_data(word)
    raw_data_str = str(raw_data) if raw_data else "No dictionary data found."
    
    enrich_prompt = load_prompt("enrichment")
    enriched_data = await structured_llm_call(
        enrich_prompt["template"],
        EnrichedWord,
        word=word,
        raw_data=raw_data_str
    )
    return enriched_data

async def command_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Command Agent ---")
    msg = state["request"].message
    
    # 1. Extract entities
    extract_prompt = load_prompt("command_extractor")
    entities = await structured_llm_call(extract_prompt["template"], CommandExtraction, user_message=msg)
    
    word_to_add = entities.word
    target_collection = entities.collection or "General"
    
    if not word_to_add:
        return {
            "response_type": "result",
            "agent_outcome": {"message": "Không tìm thấy từ nào để thêm. Vui lòng thử lại."}
        }
    
    # 2. Check collection existence (Mock logic)
    if target_collection.lower() == 'technology':
        return {
            "response_type": "confirm",
            "agent_outcome": {
                "message": f"Bạn chưa có bộ '{target_collection}'. Bạn có muốn tạo bộ này không?",
                "payload": {
                    "suggested_action": {
                        "action": "create_collection",
                        "collection_name": target_collection
                    }
                }
            }
        }
        
    # 3. Enrichment
    enriched_word = await _enrich_word_data(word_to_add)
    
    # 4. Prepare Mock API Payload for Spring Boot
    mock_api_payload = {
        "client_request_id": str(uuid.uuid4()),
        "word": enriched_word.word,
        "phonetic": enriched_word.phonetics.uk.text if enriched_word.phonetics and enriched_word.phonetics.uk else None,
        "definitions_en": [enriched_word.definition_en] if enriched_word.definition_en else [],
        "examples_en": [ex.en for ex in enriched_word.examples if ex.en],
        "definition_vi": enriched_word.definition_vi,
        "examples_vi": [ex.vi for ex in enriched_word.examples if ex.vi],
        "idioms": enriched_word.idioms,
        "tags": [target_collection],
        "added_by": state["request"].user_id,
        "source": "chatbot"
    }
    
    # 5. Persist to MongoDB (Mocking the backend call)
    word_id = await MongoService.mock_save_word(mock_api_payload)
    
    # 6. Mock Response from Backend
    mock_backend_response = {
      "status": "created",
      "wordId": word_id, # Sử dụng ID thật từ Mongo
      "collectionId": "c456_mock"
    }

    # 7. Prepare final payload for frontend
    final_payload = {
        **mock_backend_response,
        **enriched_word.model_dump()
    }

    return {
        "response_type": "result",
        "agent_outcome": {
            "message": f"Đã thêm từ '{word_to_add}' vào bộ {target_collection}.",
            "payload": final_payload
        }
    }