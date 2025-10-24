import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict, Type
from pydantic import BaseModel

def get_llm(temperature: float = 0.0):
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )

async def structured_llm_call(prompt_template: str, pydantic_model: Type[BaseModel], **kwargs) -> BaseModel:
    """Gọi LLM và ép kiểu trả về theo Pydantic model (Structured Output)."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(pydantic_model)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | structured_llm
    return await chain.ainvoke(kwargs)

async def basic_llm_call(prompt_template: str, **kwargs) -> str:
    """Gọi LLM và trả về string thông thường."""
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    result = await chain.ainvoke(kwargs)
    return result.content