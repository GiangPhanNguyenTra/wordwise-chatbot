from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.models.chat import AgentState
from app.core.tools.enrichment_tool import enrich_word_data
from app.utils.helpers import load_prompt
from app.services.core_service import core_service 
from app.core.tools.llm_tool import structured_llm_call

class AddWordExtraction(BaseModel):
    word: Optional[str] = None
    collection: Optional[str] = None

class CreateCollectionExtraction(BaseModel):
    collection: Optional[str] = None

async def _execute_add_word_to_core(jwt: str, word_to_add: str, collection_name: str) -> Dict[str, Any]:
    """
    Enriches a word and calls the core service to add it to a collection.
    This function handles the final step of the add word flow.
    """
    try:
        enriched_word = await enrich_word_data(word_to_add)
        if not enriched_word:
            return {
                "response_type": "result",
                "agent_outcome": {"message": f"Không thể tìm thấy thông tin chi tiết cho từ '{word_to_add}'."}
            }

        await core_service.add_words_to_collection(collection_name, [enriched_word], jwt)
        
        return {
            "response_type": "result",
            "intent_from_agent": "add_word",
            "agent_outcome": {
                "message": f"Đã thêm từ '{word_to_add}' vào bộ sưu tập '{collection_name}'.",
                "payload": enriched_word.model_dump()
            }
        }
    except Exception as e:
        print(f"Error calling core service to add word: {e}")
        return {
            "response_type": "error",
            "agent_outcome": {"message": "Đã xảy ra lỗi khi cố gắng thêm từ. Vui lòng thử lại."}
        }

async def handle_add_word(state: AgentState) -> Dict[str, Any]:
    """Handles the 'add_word' intent by extracting entities and interacting with the core service."""
    jwt = state["request"].jwt
    if not jwt:
        return {"response_type": "error", "agent_outcome": {"message": "Hành động này yêu cầu xác thực. Vui lòng cung cấp token."}}

    msg = state["request"].message
    extract_prompt = load_prompt("command_extractor")
    entities = await structured_llm_call(extract_prompt["template"], AddWordExtraction, user_message=msg)
    
    word_to_add = entities.word
    if not word_to_add:
        return {"response_type": "result", "agent_outcome": {"message": "Không tìm thấy từ nào để thêm trong câu của bạn."}}

    target_collection_name = entities.collection or "General"
    collection_exists = await core_service.check_collection_exists(target_collection_name, jwt)
    
    if not collection_exists:
        # Nếu collection chưa tồn tại, yêu cầu xác nhận từ người dùng
        pending_context = {
            "action": "confirm_create_and_add",
            "data": {"collection_name": target_collection_name, "word_to_add": word_to_add}
        }
        return {
            "response_type": "confirm",
            "pending_action_context": pending_context,
            "agent_outcome": {
                "message": f"Bạn chưa có bộ sưu tập '{target_collection_name}'. Bạn có muốn tạo mới và thêm từ vào không?",
                "payload": pending_context
            }
        }
    
    # Nếu collection đã tồn tại, thêm từ vào luôn
    return await _execute_add_word_to_core(jwt, word_to_add, target_collection_name)

async def handle_create_collection(state: AgentState) -> Dict[str, Any]:
    jwt = state["request"].jwt
    if not jwt:
        return {"response_type": "error", "agent_outcome": {"message": "Hành động này yêu cầu xác thực. Vui lòng cung cấp token."}}
        
    msg = state["request"].message
    
    prompt_template = """From the user's message, extract the name of the collection they want to create.
The user's message is: '{user_message}'.
Your response must be a single, valid JSON object with one key: 'collection'.
For example: {{"collection": "My New Collection"}}"""

    entities = await structured_llm_call(
        prompt_template,
        CreateCollectionExtraction,
        user_message=msg
    )

    collection_name = entities.collection
    if not collection_name:
        return {"response_type": "result", "agent_outcome": {"message": "Tôi không thể xác định được tên bộ sưu tập bạn muốn tạo."}}

    try:
        await core_service.create_collection(collection_name, jwt)
        return {
            "response_type": "result",
            "intent_from_agent": "create_collection",
            "agent_outcome": {"message": f"Đã tạo thành công bộ sưu tập '{collection_name}'."}
        }
    except Exception as e:
        print(f"Error calling core service to create collection: {e}")
        return {
            "response_type": "error",
            "agent_outcome": {"message": f"Đã xảy ra lỗi khi tạo bộ sưu tập '{collection_name}'. Có thể nó đã tồn tại."}
        }

async def command_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    The main node for handling user commands like adding words or creating collections.
    It routes to specific handlers based on intent or pending actions.
    """
    print("--- [NODE] Command Agent ---")
    message = state["request"].message
    pending_action = state.get("pending_action_context")
    intent = state.get("intent")
    jwt = state["request"].jwt

    # Xử lý hành động đang chờ xác nhận
    if pending_action:
        is_confirmation = message.lower() in ["yes", "y", "ok", "đồng ý", "tạo đi", "tạo"]
        if is_confirmation:
            action_type = pending_action.get("action")
            if action_type == "confirm_create_and_add":
                data = pending_action["data"]
                collection_name = data["collection_name"]
                word_to_add = data["word_to_add"]
                print(f"User confirmed. Creating collection '{collection_name}' and adding word '{word_to_add}'.")
                # API add-words sẽ tự tạo collection nếu chưa có
                return await _execute_add_word_to_core(jwt, word_to_add, collection_name)
        else:
            # Người dùng từ chối hoặc đưa ra một yêu cầu mới
            return {
                "response_type": "result",
                "pending_action_context": None, # Xóa context
                "agent_outcome": {"message": "Đã hủy hành động. Bạn muốn làm gì tiếp theo?"}
            }

    # Định tuyến dựa trên intent ban đầu
    if intent == "add_word":
        return await handle_add_word(state)
    elif intent == "create_collection":
        return await handle_create_collection(state)
    
    return {"response_type": "result", "agent_outcome": {"message": "Lệnh này hiện chưa được hỗ trợ."}}