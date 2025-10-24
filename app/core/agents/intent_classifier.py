from pydantic import BaseModel, Field
from app.models.chat import AgentState
from app.core.tools.llm_tool import structured_llm_call
from app.utils.helpers import load_prompt
from typing import Dict, Any

class IntentOutput(BaseModel):
    intent: str = Field(..., description="The classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0)

async def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Intent Classifier ---")
    prompt_cfg = load_prompt("intent_classifier")
    
    try:
        result = await structured_llm_call(
            prompt_cfg["template"],
            IntentOutput,
            user_message=state["request"].message
        )
        return {"intent": result.intent, "confidence": result.confidence}
    except Exception as e:
        print(f"Intent classification failed: {e}")
        return {"intent": "fallback", "confidence": 0.0}