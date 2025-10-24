from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.core.orchestrator import graph_app

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        initial_state = {"request": request, "response_type": "result"}
        
        result_state = await graph_app.ainvoke(initial_state)
        
        final_response = result_state.get("final_response")
        if not final_response:
             raise HTTPException(status_code=500, detail="Graph failed to produce a response.")
             
        return final_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))