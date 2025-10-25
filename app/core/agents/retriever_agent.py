import asyncio
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

async def _expand_query(question: str) -> str:
    print(f"--- Expanding query: '{question}' ---")
    prompt_cfg = load_prompt("query_expansion")
    expanded_query = await basic_llm_call(prompt_cfg["template"], question=question)
    clean_query = expanded_query.split('\n')[0].strip().strip('"')
    print(f"--- Expanded query: '{clean_query}' ---")
    return clean_query

async def retriever_agent_node(state: AgentState) -> dict[str, any]:
    print("--- [NODE] Optimized Parallel Retriever Agent ---")
    original_query = state["request"].message
    
    # Kỹ thuật tối ưu: Chạy song song việc làm giàu câu hỏi và tìm kiếm với câu hỏi gốc
    # Điều này giúp giảm độ trễ tổng thể.
    expanded_query_task = asyncio.create_task(_expand_query(original_query))
    original_query_vec = embed_text(original_query)
    original_search_task = asyncio.create_task(
        MongoService.vector_search("rag_documents", original_query_vec, limit=2)
    )

    # Chờ tác vụ làm giàu câu hỏi hoàn thành
    expanded_query = await expanded_query_task
    expanded_query_vec = embed_text(expanded_query)
    expanded_search_task = asyncio.create_task(
        MongoService.vector_search("rag_documents", expanded_query_vec, limit=2)
    )

    # Lấy kết quả từ cả hai lần tìm kiếm
    original_docs, expanded_docs = await asyncio.gather(
        original_search_task,
        expanded_search_task
    )

    # Gộp và loại bỏ các documents trùng lặp
    all_docs = {doc['content']: doc for doc in expanded_docs + original_docs}.values()
    
    if not all_docs:
        return {
            "response_type": "result",
            "agent_outcome": {"message": "Xin lỗi, tôi không tìm thấy tài liệu liên quan."}
        }
    
    context_str = "\n\n".join([f"Source: {d.get('source', 'N/A')}\nContent: {d.get('content', '')}" for d in all_docs])
    
    # Tạo câu trả lời với ngữ cảnh đầy đủ hơn
    rag_prompt_cfg = load_prompt("rag_prompt")
    rag_result = await structured_llm_call(
        rag_prompt_cfg["template"],
        RagOutput,
        context=context_str,
        question=original_query
    )

    payload = {
        **rag_result.model_dump(),
        "sources": list(all_docs),
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