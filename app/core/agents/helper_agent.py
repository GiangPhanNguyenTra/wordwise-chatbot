from pydantic import BaseModel
from typing import Dict, Any, List
from app.models.chat import AgentState
from app.core.tools.llm_tool import structured_llm_call
from app.utils.helpers import load_prompt

class SuggestedAction(BaseModel):
    label: str
    type: str
    value: str

class HelperOutput(BaseModel):
    message: str
    suggested_actions: List[SuggestedAction]

async def helper_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- [NODE] LLM-powered Helper Agent ---")
    prompt_cfg = load_prompt("helper_prompt")
    
    try:
        result = await structured_llm_call(prompt_cfg["template"], HelperOutput)
        
        return {
            "response_type": "result",
            "agent_outcome": {
                "message": result.message,
                "payload": {
                    "suggested_actions": [action.model_dump() for action in result.suggested_actions]
                }
            }
        }
    except Exception as e:
        print(f"Helper agent failed: {e}")
        return {
            "response_type": "result",
            "agent_outcome": {
                "message": "Tôi có thể giúp bạn thêm từ mới và trả lời các câu hỏi về ứng dụng."
            }
        }