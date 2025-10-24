from app.models.chat import AgentState
from typing import Dict, Any

async def helper_agent_node(state: AgentState) -> Dict[str, Any]:
    return {
        "agent_outcome": {
            "message": "Tôi là trợ lý học từ vựng. Bạn có thể:\n- Thêm từ mới: 'Add hello'\n- Tạo bộ từ: 'Create collection TOEIC'\n- Hỏi đáp về cách dùng ứng dụng."
        }
    }