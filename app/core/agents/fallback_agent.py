from typing import Dict, Any
from app.models.chat import AgentState

async def fallback_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Fallback Agent ---")
    intent = state.get("intent")
    
    message = ""
    if intent == "fallback":
        message = "Tôi là trợ lý ảo chuyên về Hệ thống học từ vựng Word Wise. Rất tiếc, tôi không có thông tin về các chủ đề khác. Bạn có câu hỏi nào về ứng dụng không?"
    elif intent == "smalltalk":
        message = "Chào bạn! Bạn cần tôi giúp gì về việc học từ vựng hôm nay?"
    else:
        message = "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể hỏi về các chức năng của ứng dụng hoặc yêu cầu trợ giúp."

    return {
        "response_type": "result",
        "agent_outcome": {
            "message": message
        }
    }