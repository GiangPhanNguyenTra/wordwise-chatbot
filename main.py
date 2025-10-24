from dotenv import load_dotenv
load_dotenv() # Load .env đầu tiên

from fastapi import FastAPI
from app.api import chat
from app.services.mongo_service import MongoService

app = FastAPI(title="Word Wise Chatbot")

@app.on_event("startup")
async def startup():
    MongoService.get_db()
    print("MongoDB Connected. Chatbot Ready.")

@app.on_event("shutdown")
async def shutdown():
    await MongoService.close()

app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

@app.get("/health")
def health_check():
    return {"status": "ok"}