# app/services/mongo_service.py
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
    async def vector_search(cls, collection_name: str, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Thực hiện vector search trên Atlas."""
        db = cls.get_db()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "content_embeddings",
                    "queryVector": query_vector,
                    "numCandidates": 50,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "content_embeddings": 0, # Không cần trả về vector
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        try:
            # SỬA LẠI: Tên collection khớp với cấu hình của bạn
            results = await db[collection_name].aggregate(pipeline).to_list(length=limit)
            return results
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    @classmethod
    async def mock_save_word(cls, word_data: Dict[str, Any]):
        """Tạm thời lưu trực tiếp vào Mongo để test luồng add_word."""
        db = cls.get_db()
        word_data["created_at"] = datetime.utcnow()
        result = await db.words.insert_one(word_data)
        return str(result.inserted_id)