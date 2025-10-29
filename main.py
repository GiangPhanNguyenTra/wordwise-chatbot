from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api import chat, enrichment
from app.services.mongo_service import MongoService
from app.embeddings.vector_store import get_embedding_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Word Wise Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    MongoService.get_db()
    print("MongoDB Connected.")
    get_embedding_model()
    print("Embedding model loaded. Chatbot Ready.")

@app.on_event("shutdown")
async def shutdown():
    await MongoService.close()

app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(enrichment.router, prefix="/api/v1/enrich", tags=["Enrichment"])

@app.get("/health")
def health_check():
    return {"status": "ok"}