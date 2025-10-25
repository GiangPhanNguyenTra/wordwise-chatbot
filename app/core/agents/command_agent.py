import uuid
import re
from pydantic import BaseModel
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
    raw_data = await fetch_dictionary_data(word)
    raw_data_str = str(raw_data) if raw_data else "No dictionary data found."
    enrich_prompt = load_prompt("enrichment")
    return await structured_llm_call(enrich_prompt["template"], EnrichedWord, word=word, raw_data=raw_data_str)

async def _process_add_word_flow(user_id: str, word_to_add: str, collection_name: str):
    enriched_word = await _enrich_word_data(word_to_add)
    collection = await MongoService.find_collection_by_name(user_id, collection_name)
    if not collection:
        raise ValueError(f"Collection {collection_name} not found unexpectedly.")

    document_to_save = enriched_word.model_dump()
    document_to_save["user_id"] = user_id
    document_to_save["collection_id"] = str(collection.get("_id"))
    document_to_save["collection_name"] = collection.get("name")
    
    word_id = await MongoService.save_word_to_db(document_to_save)
    
    return {
        "response_type": "result",
        "intent_from_agent": "add_word",
        "agent_outcome": {
            "message": f"Đã thêm từ '{word_to_add}' vào bộ sưu tập {collection.get('name')}.",
            "payload": {
                "wordId": word_id,
                "collectionId": str(collection.get("_id")),
                **enriched_word.model_dump()
            }
        }
    }

async def handle_stateless_add_word(state: AgentState) -> Dict[str, Any]:
    user_id = state["request"].user_id
    msg = state["request"].message
    extract_prompt = load_prompt("command_extractor")
    entities = await structured_llm_call(extract_prompt["template"], CommandExtraction, user_message=msg)
    
    target_collection_name = entities.collection or "General"
    word_to_add = entities.word

    if not word_to_add:
        return {"response_type": "result", "agent_outcome": {"message": "Không tìm thấy từ nào để thêm."}}

    collection = await MongoService.find_collection_by_name(user_id, target_collection_name)
    
    if not collection:
        pending_context = {
            "action": "confirm_create_collection",
            "data": {"collection_name": target_collection_name, "word_to_add": word_to_add}
        }
        return {
            "response_type": "confirm",
            "pending_action_context": pending_context,
            "agent_outcome": {
                "message": f"Bạn chưa có bộ sưu tập '{target_collection_name}'. Bạn có muốn tạo nó không?",
                "payload": pending_context
            }
        }
    
    return await _process_add_word_flow(user_id, word_to_add, target_collection_name)

async def command_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Command Agent ---")
    user_id = state["request"].user_id
    message = state["request"].message
    pending_action = state.get("pending_action_context")
    intent = state.get("intent")

    if pending_action:
        is_confirmation = message.lower() in ["yes", "y", "ok", "đồng ý", "tạo đi", "tạo"]
        if is_confirmation:
            action_type = pending_action.get("action")
            if action_type == "confirm_create_collection":
                data = pending_action["data"]
                collection_name = data["collection_name"]
                word_to_add = data["word_to_add"]

                print(f"User confirmed. Creating collection '{collection_name}' for user '{user_id}'...")
                await MongoService.create_collection(user_id, collection_name)
                print(f"Collection '{collection_name}' created successfully.")
                
                print(f"Proceeding to add word '{word_to_add}' to the new collection.")
                return await _process_add_word_flow(user_id, word_to_add, collection_name)
        else:
            print("User interrupted a pending action. Cancelling and processing new request.")
            if intent == "add_word":
                 return await handle_stateless_add_word(state)
            return {
                "response_type": "result",
                "agent_outcome": {"message": "Đã hủy thao tác cũ. Xin mời bạn đưa ra yêu cầu mới."}
            }

    # Luồng xử lý stateless bình thường
    if intent == "add_word":
        return await handle_stateless_add_word(state)
    
    return {"response_type": "result", "agent_outcome": {"message": "Lệnh này hiện chưa được hỗ trợ."}}