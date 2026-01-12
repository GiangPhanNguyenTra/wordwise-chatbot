from typing import Dict, Any
from app.models.chat import AgentState

async def fallback_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Fallback Agent ---")
    intent = state.get("intent")
    
    # Định nghĩa response song ngữ
    answer_en = ""
    answer_vi = ""
    
    if intent == "fallback":
        answer_en = "I am a virtual assistant specializing in the Word Wise Vocabulary System. Unfortunately, I do not have information on other topics. Do you have any questions about the application?"
        answer_vi = "Tôi là trợ lý ảo chuyên về Hệ thống học từ vựng Word Wise. Rất tiếc, tôi không có thông tin về các chủ đề khác. Bạn có câu hỏi nào về ứng dụng không?"
    elif intent == "smalltalk":
        answer_en = "Hello! How can I help you with your vocabulary learning today?"
        answer_vi = "Chào bạn! Bạn cần tôi giúp gì về việc học từ vựng hôm nay?"
    else:
        answer_en = "Sorry, I didn't understand that. You can ask about app features or request assistance."
        answer_vi = "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể hỏi về các chức năng của ứng dụng hoặc yêu cầu trợ giúp."

    # Cấu trúc giống RagOutput
    payload = {
        "answer_en": answer_en,
        "answer_vi": answer_vi,
        "detail": answer_vi, # Có thể để giống answer_vi hoặc mô tả thêm
        "confidence": 1.0,   # Hardcode vì đây là câu trả lời tĩnh
        "sources": [],       # Không có source
        "suggested_actions": [] 
    }

    return {
        "response_type": "result",
        "agent_outcome": {
            "message": answer_vi, # Message chính hiển thị
            "payload": payload
        }
    }