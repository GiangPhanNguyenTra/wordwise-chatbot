from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, TypedDict, Union

# --- API Models (Khớp 100% với spec) ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    jwt: Optional[str] = None

class ChatResponse(BaseModel):
    status: str = "ok"
    session_id: str
    type: str = Field(..., description="'result', 'confirm', or 'clarify'")
    intent: str
    payload: Optional[Dict[str, Any]] = None
    message: str

# --- LangGraph State ---
class AgentState(TypedDict):
    request: ChatRequest
    intent: str
    confidence: float
    # Dữ liệu tạm thời được các node xử lý và truyền đi
    agent_outcome: Optional[Dict[str, Any]]
    # Loại response cuối cùng sẽ được format
    response_type: str
    # Response cuối cùng sau khi format
    final_response: Optional[ChatResponse]