from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.core.orchestrator import graph_app
from app.services.mongo_service import MongoService

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        history = await MongoService.get_conversation_history(request.session_id)
        pending_action = await MongoService.get_pending_action(request.session_id)
        
        initial_state = {
            "request": request,
            "memory": history,
            "pending_action_context": pending_action,
            "response_type": "result",
            "intent_from_agent": None,
        }
        
        result_state = await graph_app.ainvoke(initial_state)
        
        final_response = result_state.get("final_response")
        if not final_response:
             raise HTTPException(status_code=500, detail="Graph failed to produce a response.")
             
        new_messages = [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": final_response.message}
        ]
        
        new_pending_context = result_state.get("pending_action_context")
        
        await MongoService.append_to_conversation(
            request.session_id, 
            request.user_id, 
            new_messages,
            new_pending_context
        )
             
        return final_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))