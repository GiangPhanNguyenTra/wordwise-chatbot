from typing import Dict, Any
from app.models.chat import AgentState

STATIC_HELPER_MESSAGE = "Chào mừng bạn đến với Word Wise! Bạn có thể thêm từ vựng mới, hỏi về tính năng của ứng dụng hoặc tạo bộ từ vựng mới."
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
    
    return {
        "response_type": "result",
        "agent_outcome": {
            "message": STATIC_HELPER_MESSAGE,
            "payload": {
                "suggested_actions": STATIC_SUGGESTED_ACTIONS
            }
        }
    }