# app/core/agents/intent_classifier.py
from pydantic import BaseModel, Field
from typing import Dict, Any
from app.models.chat import AgentState
from app.core.tools.llm_tool import structured_llm_call
from app.utils.helpers import load_prompt

class IntentOutput(BaseModel):
    intent: str = Field(..., description="The classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0)

async def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Intent Classifier ---")
    
    # Sử dụng lại prompt 'intent_classifier_v2' đã được cải thiện
    prompt_cfg = load_prompt("intent_classifier") 
    
    try:
        # Gọi hàm structured_llm_call đã được sửa lỗi
        result = await structured_llm_call(
            prompt_cfg["template"],
            IntentOutput,
            user_message=state["request"].message
        )
        print(f"Intent classified: {result.intent} with confidence {result.confidence}")
        return {"intent": result.intent, "confidence": result.confidence}
    except Exception as e:
        # Ghi log lỗi và trả về fallback
        print(f"Intent classification failed: {e}")
        return {"intent": "fallback", "confidence": 0.0}