# scripts/embed_data.py
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

import docx
import pypdf

# --- Cấu hình ---
DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "documents"
MONGO_COLLECTION_NAME = "rag_documents"

# --- Hàm trích xuất văn bản (giữ nguyên) ---
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

async def embed_and_load_data():
    print("Starting data embedding and loading process...")
    
    # --- THAY ĐỔI QUAN TRỌNG: Đảm bảo load đúng file .env ---
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"CRITICAL: .env file not found at path: {env_path}")
    load_dotenv(dotenv_path=env_path)
    print(f".env file loaded from: {env_path}")
    
    # --- THAY ĐỔI QUAN TRỌNG: "Sanity Check" ---
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    print(f"DEBUG: Model name from .env is: '{model_name}'") # Dòng debug
    if not model_name or model_name.strip() == "":
        raise ValueError("CRITICAL: EMBEDDING_MODEL_NAME is not set in your .env file!")
    # --- KẾT THÚC THAY ĐỔI ---
    
    # 1. Kết nối MongoDB ... (giữ nguyên)
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_uri or not db_name:
        raise ValueError("MONGO_URI and MONGO_DB_NAME must be set in .env")
    
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[MONGO_COLLECTION_NAME]

    print(f"Clearing old data from '{MONGO_COLLECTION_NAME}' collection...")
    await collection.delete_many({})
    print("Old data cleared.")

    # 2. Tải model embedding
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    # ... Phần còn lại của script giữ nguyên ...
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    if not DOCUMENTS_PATH.exists():
        DOCUMENTS_PATH.mkdir()
        print(f"Warning: Directory '{DOCUMENTS_PATH}' created. Please add documents.")
        return
    documents_to_insert: List[Dict[str, Any]] = []
    files_to_process = list(DOCUMENTS_PATH.glob("*"))
    file_handlers = {".docx": extract_text_from_docx, ".pdf": extract_text_from_pdf, ".txt": extract_text_from_txt, ".md": extract_text_from_txt}
    
    for file_path in files_to_process:
        file_ext = file_path.suffix.lower()
        if file_ext in file_handlers:
            print(f"\nProcessing file: {file_path.name}...")
            handler = file_handlers[file_ext]
            content = handler(file_path)
            if not content:
                print(f"No text extracted from {file_path.name}. Skipping.")
                continue
            chunks = text_splitter.split_text(content)
            print(f"Split into {len(chunks)} chunks.")
            chunk_embeddings = model.encode(chunks, show_progress_bar=True)
            for i, chunk_text in enumerate(chunks):
                document = {"source": file_path.name, "content": chunk_text, "chunk_id": i + 1, "content_embeddings": chunk_embeddings[i].tolist()}
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