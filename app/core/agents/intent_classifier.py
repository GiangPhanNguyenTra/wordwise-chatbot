from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from app.models.chat import AgentState
from app.core.tools.llm_tool import structured_llm_call
from app.utils.helpers import load_prompt

class IntentOutput(BaseModel):
    intent: str = Field(..., description="The classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0)

INTENT_KEYWORDS_STRICT = {
    "add_word": ["add", "thêm", "save", "lưu từ", "add the word"],
    "create_collection": ["create", "tạo bộ", "new collection"],
    "help": ["help", "trợ giúp", "menu"],
}

def strict_rule_based_classifier(message: str) -> Optional[Dict[str, Any]]:
    msg_lower = message.lower().strip()
    
    for intent, keywords in INTENT_KEYWORDS_STRICT.items():
        if any(keyword in msg_lower for keyword in keywords):
            return {"intent": intent, "confidence": 1.0}

    return None

async def llm_classifier_for_dialogue(message: str) -> Dict[str, Any]:
    print("--- Rules did not match. Falling back to LLM Classifier for dialogue analysis ---")
    prompt_cfg = load_prompt("intent_classifier")
    
    try:
        result = await structured_llm_call(
            prompt_cfg["template"],
            IntentOutput,
            user_message=message
        )
        return {"intent": result.intent, "confidence": result.confidence}
    except Exception as e:
        print(f"LLM Intent classification failed: {e}")
        return {"intent": "fallback", "confidence": 0.0}

async def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Hybrid Intent Classifier v2 ---")
    user_message = state["request"].message
    
    command_result = strict_rule_based_classifier(user_message)
    
    if command_result:
        print(f"Intent classified by STRICT RULES: {command_result['intent']}")
        return command_result
    
    dialogue_result = await llm_classifier_for_dialogue(user_message)
    print(f"Intent classified by LLM: {dialogue_result['intent']} with confidence {dialogue_result['confidence']}")
    return dialogue_result