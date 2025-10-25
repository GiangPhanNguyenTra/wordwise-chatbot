# app/core/agents/retriever_agent.py
from pydantic import BaseModel, Field
from app.models.chat import AgentState
from app.embeddings.vector_store import embed_text
from app.services.mongo_service import MongoService
from app.core.tools.llm_tool import structured_llm_call, basic_llm_call
from app.utils.helpers import load_prompt

class RagOutput(BaseModel):
    answer_en: str
    answer_vi: str
    detail: str
    confidence: float = Field(..., ge=0.0, le=1.0)

async def _generate_hypothetical_document(question: str) -> str:
    """Sử dụng LLM để tạo ra một tài liệu giả định trả lời câu hỏi."""
    print(f"--- Generating hypothetical document for question: '{question}' ---")
    prompt_cfg = load_prompt("hyde_prompt")
    hypothetical_doc = await basic_llm_call(prompt_cfg["template"], question=question)
    print(f"--- Hypothetical document generated: '{hypothetical_doc[:150]}...' ---")
    return hypothetical_doc

async def retriever_agent_node(state: AgentState) -> dict[str, any]:
    print("--- [NODE] Retriever Agent (RAG) ---")
    original_query = state["request"].message
    
    # BƯỚC 1: TẠO TÀI LIỆU GIẢ ĐỊNH (HyDE)
    hypothetical_doc = await _generate_hypothetical_document(original_query)
    
    # BƯỚC 2: TẠO EMBEDDING TỪ TÀI LIỆU GIẢ ĐỊNH
    query_vec = embed_text(hypothetical_doc)
    
    # BƯỚC 3: TÌM KIẾM VECTOR
    docs = await MongoService.vector_search("rag_documents", query_vec, limit=3)
    
    if not docs:
        return {
            "response_type": "result",
            "agent_outcome": {"message": "Xin lỗi, tôi không tìm thấy tài liệu liên quan."}
        }
    
    context_str = "\n\n".join([f"Source: {d.get('source', 'N/A')}\nContent: {d.get('content', '')}" for d in docs])
    
    # BƯỚC 4: TẠO CÂU TRẢ LỜI TỪ NGỮ CẢNH TÌM ĐƯỢC
    rag_prompt_cfg = load_prompt("rag_prompt")
    rag_result = await structured_llm_call(
        rag_prompt_cfg["template"],
        RagOutput,
        context=context_str,
        question=original_query
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