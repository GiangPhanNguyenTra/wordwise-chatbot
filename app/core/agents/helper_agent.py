from typing import Dict, Any
from app.models.chat import AgentState

# Song ngữ cho Helper
STATIC_HELPER_MSG_VI = "Chào mừng bạn đến với Word Wise Chatbot! Bạn có thể thêm từ vựng mới, hỏi về tính năng của ứng dụng hoặc tạo bộ từ vựng mới."
STATIC_HELPER_MSG_EN = "Welcome to Word Wise Chatbot! You can add new vocabulary, ask about app features, or create new vocabulary sets."

STATIC_SUGGESTED_ACTIONS = [
    {
        "label": "Thêm một từ vựng mới",
        "type": "user_input",
        "value": "Thêm từ 'serendipity' vào bộ General"
    },
    {
        "label": "Cách lặp lại ngắt quãng hoạt động?",
        "type": "user_input",
        "value": "Lặp lại ngắt quãng hoạt động như thế nào?"
    },
    {
        "label": "Tạo một bộ từ vựng",
        "type": "user_input",
        "value": "Tạo bộ từ vựng cho chủ đề 'Travel'"
    }
]

async def helper_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] Rule-based Helper Agent ---")
    
    # Giả lập cấu trúc giống RagOutput
    payload = {
        "answer_en": STATIC_HELPER_MSG_EN,
        "answer_vi": STATIC_HELPER_MSG_VI,
        "detail": STATIC_HELPER_MSG_VI,
        "confidence": 1.0,
        "sources": [],
        "suggested_actions": STATIC_SUGGESTED_ACTIONS
    }

    return {
        "response_type": "result",
        "agent_outcome": {
            "message": STATIC_HELPER_MSG_VI,
            "payload": payload
        }
    }