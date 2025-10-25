from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from app.models.chat import AgentState
from app.core.tools.llm_tool import structured_llm_call
from app.utils.helpers import load_prompt

class IntentOutput(BaseModel):
    intent: str = Field(..., description="The classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0)

# 1. intent classified (RULE-BASED)
INTENT_KEYWORDS = {
    "help": ["help", "trợ giúp", "menu", "hướng dẫn", "cần giúp"],
    "add_word": ["add", "add the word" , "thêm", "save", "lưu từ"],
    "create_collection": ["create", "tạo bộ", "new collection"],
}

def hybrid_rule_based_classifier(message: str) -> Optional[Dict[str, Any]]:
    msg_lower = message.lower().strip()
    # Ưu tiên các từ khóa của 'help' và các lệnh tạo/lưu
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in msg_lower for keyword in keywords):
            return {"intent": intent, "confidence": 1.0}
    
    if "làm được gì" in msg_lower or "chức năng" in msg_lower:
        return {"intent": "help", "confidence": 1.0}

    return None

# 2.  intent classified (LLM) 
async def llm_classifier(message: str) -> Dict[str, Any]:
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
    user_message = state["request"].message
    
    rule_based_result = hybrid_rule_based_classifier(user_message)
    
    if rule_based_result:
        # Nếu Rule-based thành công, dùng ngay kết quả
        return rule_based_result
    
    llm_result = await llm_classifier(user_message)
    return llm_result