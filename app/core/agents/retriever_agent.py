# app/core/agents/retriever_agent.py
from pydantic import BaseModel, Field
from app.models.chat import AgentState
from app.embeddings.vector_store import embed_text
from app.services.mongo_service import MongoService
# Sửa lại import
from app.core.tools.llm_tool import structured_llm_call
from app.utils.helpers import load_prompt

# Thêm Pydantic model cho output của RAG
class RagOutput(BaseModel):
    answer_en: str
    answer_vi: str
    detail: str
    confidence: float = Field(..., ge=0.0, le=1.0)

async def retriever_agent_node(state: AgentState) -> dict[str, any]:
    print("--- [NODE] Retriever Agent (RAG) ---")
    query = state["request"].message
    
    query_vec = embed_text(query)
    docs = await MongoService.vector_search("rag_documents", query_vec, limit=3)
    
    if not docs:
        return {
            "response_type": "result",
            "agent_outcome": {"message": "Xin lỗi, tôi không tìm thấy tài liệu liên quan."}
        }
    
    context_str = "\n\n".join([f"Source: {d.get('source', 'N/A')}\nContent: {d.get('content', '')}" for d in docs])
    
    # Sử dụng structured_llm_call để có output JSON đáng tin cậy
    rag_prompt_cfg = load_prompt("rag_prompt")
    rag_result = await structured_llm_call(
        rag_prompt_cfg["template"],
        RagOutput,
        context=context_str,
        question=query
    )

    payload = {
        **rag_result.model_dump(),
        "sources": docs,
        "suggested_actions": [
            {"label": "Xem hướng dẫn chi tiết", "type": "open_url", "url": "/help/docs"},
        ]
    }
    
    return {
        "response_type": "result",
        "agent_outcome": {
            "message": rag_result.answer_vi,
            "payload": payload
        }
    }