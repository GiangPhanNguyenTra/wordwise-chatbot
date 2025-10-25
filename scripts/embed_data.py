# scripts/embed_data.py
import os
import re
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

import docx
import pypdf

DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "documents"
MONGO_COLLECTION_NAME = "rag_documents"

def extract_text_from_docx(file_path: Path) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading docx file {file_path.name}: {e}")
        return ""

def extract_text_from_pdf(file_path: Path) -> str:
    try:
        reader = pypdf.PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error reading pdf file {file_path.name}: {e}")
        return ""

def extract_text_from_txt(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {file_path.name}: {e}")
        return ""

def parse_document_structure(text: str) -> List[Dict[str, str]]:
    """
    Tự động phân tích cấu trúc tài liệu dựa trên các tiêu đề (ví dụ: 'PHẦN 1:', '1.1.').
    Trả về một danh sách các section, mỗi section có 'title' và 'content'.
    """
    # Regex để tìm các dòng trông giống như tiêu đề
    # Mẫu này tìm các dòng bắt đầu bằng 'PHẦN X:', 'X.Y.', 'X.Y.Z.', hoặc các dòng viết hoa toàn bộ
    title_pattern = re.compile(r"^(PHẦN \d+:.*|^\d\.\d(?:\.\d)?\..*|^[A-Z\s\d:_-]{5,100}$)", re.MULTILINE)
    
    sections = []
    last_end = 0
    current_title = "Giới thiệu chung"

    for match in title_pattern.finditer(text):
        start, end = match.span()
        
        # Lấy nội dung từ tiêu đề trước đến tiêu đề này
        content_before = text[last_end:start].strip()
        if content_before:
            sections.append({"title": current_title, "content": content_before})
            
        current_title = match.group(1).strip()
        last_end = end
        
    # Thêm phần nội dung cuối cùng
    final_content = text[last_end:].strip()
    if final_content:
        sections.append({"title": current_title, "content": final_content})
        
    # Nếu không tìm thấy cấu trúc nào, coi toàn bộ tài liệu là một section
    if not sections:
        return [{"title": "Nội dung chính", "content": text}]
        
    return sections

async def embed_and_load_data():
    print("Starting data embedding and loading process...")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"CRITICAL: .env file not found at path: {env_path}")
    load_dotenv(dotenv_path=env_path)
    
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not model_name:
        raise ValueError("CRITICAL: EMBEDDING_MODEL_NAME is not set in .env!")

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_uri or not db_name:
        raise ValueError("MONGO_URI and MONGO_DB_NAME must be set in .env")
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[MONGO_COLLECTION_NAME]

    print(f"Clearing old data from '{MONGO_COLLECTION_NAME}'...")
    await collection.delete_many({})
    print("Old data cleared.")

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    if not DOCUMENTS_PATH.exists():
        DOCUMENTS_PATH.mkdir()
        print(f"Warning: Directory '{DOCUMENTS_PATH}' created. Please add documents.")
        return

    documents_to_insert = []
    files_to_process = list(DOCUMENTS_PATH.glob("*"))
    file_handlers = {".docx": extract_text_from_docx, ".pdf": extract_text_from_pdf, ".txt": extract_text_from_txt, ".md": extract_text_from_txt}
    
    for file_path in files_to_process:
        file_ext = file_path.suffix.lower()
        if file_ext in file_handlers:
            print(f"\nProcessing file: {file_path.name}...")
            handler = file_handlers[file_ext]
            full_content = handler(file_path)
            if not full_content:
                print(f"No text extracted. Skipping.")
                continue

            # TỐI ƯU: Phân tích tài liệu thành các section
            sections = parse_document_structure(full_content)
            print(f"Document parsed into {len(sections)} sections.")

            for section in sections:
                section_title = section["title"]
                section_content = section["content"]
                
                # Chia nhỏ nội dung của từng section
                chunks = text_splitter.split_text(section_content)
                
                for i, chunk_text in enumerate(chunks):
                    # TỐI ƯU: Thêm ngữ cảnh tiêu đề vào nội dung TRƯỚC KHI embedding
                    # Kỹ thuật này giúp vector chứa thông tin về section của nó.
                    content_for_embedding = f"Tiêu đề: {section_title}\n\nNội dung: {chunk_text}"
                    
                    document = {
                        "document_title": file_path.stem, # Tên file không có đuôi
                        "section_title": section_title,
                        "content": chunk_text, # Lưu nội dung gốc, không có tiêu đề
                        "source": file_path.name,
                        "chunk_id": i + 1,
                        "content_embeddings": model.encode(content_for_embedding).tolist()
                    }
                    documents_to_insert.append(document)
        else:
            print(f"\nUnsupported file type: {file_path.name}. Skipping.")

    if documents_to_insert:
        print(f"\nInserting {len(documents_to_insert)} documents into MongoDB...")
        await collection.insert_many(documents_to_insert)
        print("Data insertion complete!")
    else:
        print("No processable documents found to insert.")
    
    client.close()
    print("\nProcess finished successfully.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(embed_and_load_data())