# app/core/orchestrator.py
from langgraph.graph import StateGraph, END
from app.models.chat import AgentState, ChatResponse
from app.core.agents import (
    intent_classifier, command_agent, retriever_agent, 
    helper_agent, fallback_agent
)

def route_logic(state: AgentState) -> str:
    if state.get("pending_action_context"):
        print("--- ROUTING: Detected pending action -> To Command Agent ---")
        return "command"

    intent = state.get("intent")
    conf = state.get("confidence", 0.0)
    print(f"--- ROUTING: By intent '{intent}' with confidence {conf} ---")
    
    if intent == "app_question":
        return "retriever"
    elif intent in ["add_word", "create_collection"]:
        return "command"
    elif intent == "help":
        return "helper"
    else:
        return "fallback"

async def clarification_node(state: AgentState) -> dict:
    print("--- [NODE] Clarification ---")
    return {
        "response_type": "clarify",
        "agent_outcome": {
            "message": "Tôi chưa chắc đã hiểu ý bạn. Bạn có muốn:",
            "payload": {
                "suggested_actions": [
                    {"label": "Thêm từ mới", "intent": "add_word"},
                    {"label": "Hỏi đáp", "intent": "app_question"},
                ]
            }
        }
    }

def response_formatter_node(state: AgentState) -> dict:
    print("--- [NODE] Response Formatter ---")
    
    final_intent = state.get("intent_from_agent") or state.get("intent", "unknown")
    outcome = state.get("agent_outcome", {})
    
    final_res = ChatResponse(
        session_id=state["request"].session_id,
        intent=final_intent,
        type=state.get("response_type", "result"),
        message=outcome.get("message", "Đã xảy ra lỗi."),
        payload=outcome.get("payload")
    )
    return {"final_response": final_res}

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
    
    workflow.add_conditional_edges("classifier", route_logic, {
        "command": "command",
        "retriever": "retriever",
        "helper": "helper",
        "fallback": "fallback",
        "clarification": "clarification"
    })
    
    for node in ["command", "retriever", "helper", "fallback", "clarification"]:
        workflow.add_edge(node, "formatter")
        
    workflow.add_edge("formatter", END)
    
    return workflow.compile()

graph_app = build_graph()