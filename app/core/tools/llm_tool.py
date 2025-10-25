import os
import json
import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError
from typing import Type

def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise ValueError("LLM_MODEL is not set in your .env file!")

    print(f"--- Initializing LLM Provider: {provider.upper()} with model: {model_name} ---")

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=api_key)
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}.")

def _extract_json_from_response(text: str) -> dict:
    # Thử trích xuất JSON từ ```json
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = text.strip()

    # Làm sạch JSON: Loại bỏ ký tự không mong muốn (e.g., xuống dòng thừa, tab)
    json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)  # Loại bỏ ký tự điều khiển
    json_str = json_str.replace('\n', '').replace('\t', '')  # Loại bỏ xuống dòng và tab

    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Raw response was: {text}")
        # Thử sửa lỗi cú pháp cơ bản (e.g., thêm dấu phẩy bị thiếu)
        try:
            # Tách thành các dòng và thêm dấu phẩy giữa các phần tử
            lines = [line.strip() for line in json_str.split('}') if line.strip()]
            cleaned_json = '},'.join(lines[:-1]) + '}' if lines else '{}'
            return json.loads(cleaned_json, strict=False)
        except json.JSONDecodeError as e2:
            print(f"Failed to repair JSON: {e2}")
            raise ValueError("Failed to parse JSON from LLM response after cleanup.")

async def structured_llm_call(prompt_template: str, pydantic_model: Type[BaseModel], **kwargs) -> BaseModel:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    raw_response = await chain.ainvoke(kwargs)
    response_content = raw_response.content
    
    json_data = _extract_json_from_response(response_content)
    
    try:
        validated_data = pydantic_model.model_validate(json_data)
        return validated_data
    except ValidationError as e:
        print(f"Pydantic validation failed for model {pydantic_model.__name__}: {e}")
        print(f"JSON data received from LLM was: {json_data}")
        raise ValueError("LLM response did not match the Pydantic schema.")

async def basic_llm_call(prompt_template: str, **kwargs) -> str:
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    result = await chain.ainvoke(kwargs)
    return result.content