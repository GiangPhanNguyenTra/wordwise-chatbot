# app/core/agents/command_agent.py
import uuid
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.models.chat import AgentState
from app.models.word import EnrichedWord
from app.core.tools.llm_tool import structured_llm_call
from app.core.tools.dictionary_tool import fetch_dictionary_data
from app.utils.helpers import load_prompt
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

# --- Logic Mock để kiểm tra Collection ---
# Giả lập rằng user "u123" chỉ có collection "Personal" và "General"
MOCK_USER_COLLECTIONS = {
    "u123": ["personal", "general"]
}

async def check_collection_exists(user_id: str, collection_name: str) -> bool:
    """Mock function to check if a collection exists for a user."""
    user_collections = MOCK_USER_COLLECTIONS.get(user_id, [])
    return collection_name.lower() in user_collections

async def command_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Command Agent ---")
    msg = state["request"].message
    user_id = state["request"].user_id
    
    # 1. Extract entities
    extract_prompt = load_prompt("command_extractor")
    entities = await structured_llm_call(extract_prompt["template"], CommandExtraction, user_message=msg)
    
    word_to_add = entities.word
    # Nếu không chỉ định collection, mặc định là "General"
    target_collection = entities.collection or "General"
    
    if not word_to_add:
        return {
            "response_type": "result",
            "agent_outcome": {"message": "Không tìm thấy từ nào để thêm. Vui lòng thử lại."}
        }

    # 2. Check collection existence
    collection_exists = await check_collection_exists(user_id, target_collection)
    
    if not collection_exists:
        # Nếu collection không tồn tại, trả về response 'confirm'
        print(f"Collection '{target_collection}' not found for user '{user_id}'. Returning 'confirm'.")
        return {
            "response_type": "confirm",
            "agent_outcome": {
                "message": f"Bạn chưa có bộ sưu tập '{target_collection}'. Bạn có muốn tạo nó không?",
                "payload": {
                    "suggested_action": {
                        "action": "create_collection",
                        "collection_name": target_collection
                    }
                }
            }
        }

    # 3. Enrichment (Nếu collection đã tồn tại)
    print(f"Collection '{target_collection}' found. Proceeding with enrichment for '{word_to_add}'.")
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
        "added_by": user_id,
        "source": "chatbot"
    }
    
    # 5. Persist to MongoDB (Mocking the backend call)
    word_id = await MongoService.mock_save_word(mock_api_payload)
    print(f"Word '{word_to_add}' mock-saved to MongoDB with ID: {word_id}")
    
    # 6. Mock Response from Backend
    mock_backend_response = {
      "status": "created",
      "wordId": word_id,
      "collectionId": "c456_mock" # ID của collection 'Personal'
    }

    # 7. Prepare final payload cho Frontend
    final_payload = {
        **mock_backend_response,
        **enriched_word.model_dump()
    }

    return {
        "response_type": "result",
        "agent_outcome": {
            "message": f"Đã thêm từ '{word_to_add}' vào bộ sưu tập {target_collection}.",
            "payload": final_payload
        }
    }