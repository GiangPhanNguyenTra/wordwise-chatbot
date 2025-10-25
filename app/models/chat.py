# app/models/chat.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, TypedDict

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    jwt: Optional[str] = None

class ChatResponse(BaseModel):
    status: str = "ok"
    session_id: str
    type: str
    intent: str
    payload: Optional[Dict[str, Any]] = None
    message: str

class AgentState(TypedDict):
    request: ChatRequest
    memory: List[Dict[str, str]]
    pending_action_context: Optional[Dict[str, Any]]
    intent: str 
    intent_from_agent: Optional[str] 
    confidence: float
    agent_outcome: Optional[Dict[str, Any]]
    response_type: str
    final_response: Optional[ChatResponse]