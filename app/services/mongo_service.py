import os
import motor.motor_asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

class MongoService:
    _client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
    _db = None

    @classmethod
    def get_db(cls):
        if cls._client is None:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise ValueError("MONGO_URI not set in .env")
            cls._client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            cls._db = cls._client[os.getenv("MONGO_DB_NAME", "vocab_chatbot")]
        return cls._db

    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
    
    @classmethod
    async def find_collection_by_name(cls, user_id: str, collection_name: str) -> Optional[Dict[str, Any]]:
        db = cls.get_db()
        return await db.collections.find_one({
            "userId": user_id,
            "name": {"$regex": f"^{collection_name}$", "$options": "i"}
        })

    @classmethod
    async def create_collection(cls, user_id: str, collection_name: str) -> Dict[str, Any]:
        db = cls.get_db()
        new_collection = {
            "userId": user_id,
            "name": collection_name,
            "createdAt": datetime.utcnow()
        }
        result = await db.collections.insert_one(new_collection)
        new_collection["_id"] = result.inserted_id
        return new_collection

    @classmethod
    async def save_word_to_db(cls, word_document: Dict[str, Any]):
        db = cls.get_db()
        word_document["created_at"] = datetime.utcnow()
        result = await db.words.insert_one(word_document)
        return str(result.inserted_id)

    @classmethod
    async def get_conversation_history(cls, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        db = cls.get_db()
        convo = await db.conversations.find_one({"session_id": session_id})
        if convo and "messages" in convo:
            return convo["messages"][-limit:]
        return []

    @classmethod
    async def append_to_conversation(cls, session_id: str, user_id: str, messages: List[Dict[str, str]], pending_context: Optional[Dict] = None):
        db = cls.get_db()
        update_doc = {
            "$push": {"messages": {"$each": messages}},
            "$setOnInsert": {"user_id": user_id, "created_at": datetime.utcnow()}
        }
        if pending_context:
            update_doc["$set"] = {"pending_action_context": pending_context}
        else:
            update_doc["$unset"] = {"pending_action_context": ""}
        
        await db.conversations.update_one(
            {"session_id": session_id},
            update_doc,
            upsert=True
        )

    @classmethod
    async def get_pending_action(cls, session_id: str) -> Optional[Dict[str, Any]]:
        db = cls.get_db()
        convo = await db.conversations.find_one({"session_id": session_id})
        if convo:
            return convo.get("pending_action_context")
        return None