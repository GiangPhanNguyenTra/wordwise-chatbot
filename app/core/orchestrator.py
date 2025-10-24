from langgraph.graph import StateGraph, END
from app.models.chat import AgentState, ChatResponse
from app.core.agents import (
    intent_classifier, command_agent, retriever_agent, 
    helper_agent, fallback_agent
)

# --- Routing Logic ---
def route_intent(state: AgentState) -> str:
    intent = state.get("intent")
    conf = state.get("confidence", 0.0)
    
    if conf < 0.6: # Confidence thấp -> làm rõ
        return "clarification"
        
    if intent in ["add_word", "create_collection"]:
        return "command"
    elif intent == "question":
        return "retriever"
    elif intent == "help":
        return "helper"
    return "fallback"

# --- Graph Nodes ---
async def clarification_node(state: AgentState) -> dict:
    """Node để xử lý khi intent không chắc chắn."""
    print("--- [NODE] Clarification ---")
    return {
        "response_type": "clarify",
        "agent_outcome": {
            "message": "Tôi chưa chắc đã hiểu ý bạn. Bạn có muốn:",
            "payload": {
                "suggested_actions": [
                    {"label": "Thêm từ mới", "intent": "add_word"},
                    {"label": "Hỏi đáp", "intent": "question"},
                ]
            }
        }
    }

def response_formatter_node(state: AgentState) -> dict:
    """Node cuối cùng để format response theo đúng chuẩn API."""
    print("--- [NODE] Response Formatter ---")
    outcome = state.get("agent_outcome", {})
    
    final_res = ChatResponse(
        session_id=state["request"].session_id,
        intent=state.get("intent", "unknown"),
        type=state.get("response_type", "result"),
        message=outcome.get("message", "Đã xảy ra lỗi."),
        payload=outcome.get("payload")
    )
    return {"final_response": final_res}

# --- Graph Construction ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classifier", intent_classifier.intent_classifier_node)
    workflow.add_node("command", command_agent.command_agent_node)
    workflow.add_node("retriever", retriever_agent.retriever_agent_node)
    workflow.add_node("helper", helper_agent.helper_agent_node)
    workflow.add_node("fallback", fallback_agent.fallback_agent_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("formatter", response_formatter_node)
    
    workflow.set_entry_point("classifier")
    
    workflow.add_conditional_edges(
        "classifier",
        route_intent,
        {
            "command": "command",
            "retriever": "retriever",
            "helper": "helper",
            "fallback": "fallback",
            "clarification": "clarification"
        }
    )
    
    # Tất cả các node xử lý chính đều dẫn đến formatter
    for node in ["command", "retriever", "helper", "fallback", "clarification"]:
        workflow.add_edge(node, "formatter")
        
    workflow.add_edge("formatter", END)
    
    return workflow.compile()

graph_app = build_graph()