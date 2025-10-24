from app.models.chat import AgentState
from typing import Dict, Any

async def fallback_agent_node(state: AgentState) -> Dict[str, Any]:
    return {
        "agent_outcome": {
            "message": "Xin lỗi, tôi chưa hiểu ý bạn. Bạn có thể diễn đạt lại được không?"
        }
    }