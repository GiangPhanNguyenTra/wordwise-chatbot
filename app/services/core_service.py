import os
import httpx
from typing import List, Dict, Any, Optional

from app.models.word import EnrichedWord

class CoreService:
    def __init__(self):
        self.base_url = os.getenv("SPRING_BOOT_API_URL")
        if not self.base_url:
            raise ValueError("SPRING_BOOT_API_URL environment variable not set.")

    async def _make_request(self, method: str, endpoint: str, jwt: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> httpx.Response:
        headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}
        
        base = self.base_url.rstrip("/")
        path = endpoint.lstrip("/")
        
        full_url = f"{base}/{path}"
    
        print(f"--- CORE SERVICE CALL: {method} {full_url}")
        
        async with httpx.AsyncClient() as client:
            res = await client.request(
                method, 
                full_url, 
                headers=headers, 
                params=params, 
                json=json_data,
                timeout=10.0 
            )
            res.raise_for_status()
            return res
        
    async def check_collection_exists(self, collection_name: str, jwt: str) -> bool:
        """Checks if a collection exists for the user."""
        try:
            response = await self._make_request("GET", "/api/v1/collections/exists", jwt, params={"collectionName": collection_name})
            data = response.json()
            return data.get("data", {}).get("exists", False)
        except httpx.HTTPStatusError as e:
            print(f"Error checking collection existence: {e}")
            return False

    async def add_words_to_collection(self, collection_name: str, words: List[EnrichedWord], jwt: str) -> Dict[str, Any]:
        """Adds words to a collection, creating it if it doesn't exist."""
        payload = {
            "collection": collection_name,
            "words": [word.model_dump() for word in words]
        }
        response = await self._make_request("POST", "/api/v1/collections/add-words", jwt, json_data=payload)
        return response.json()
        
    async def create_collection(self, collection_name: str, jwt: str) -> Dict[str, Any]:
        """Creates a new empty collection for the user."""
        payload = {"name": collection_name, "description": " "}
        response = await self._make_request("POST", "/api/v1/collections", jwt, json_data=payload)
        return response.json()
    
    async def get_collection_names(self, jwt: str) -> List[str]:
        """Retrieves a list of collection names for the authenticated user."""
        try:
            response = await self._make_request("GET", "/api/v1/collections/names", jwt)
            data = response.json()
            return data.get("data", [])
        except httpx.HTTPStatusError as e:
            print(f"Error fetching collection names: {e}")
            return []

# Singleton instance
core_service = CoreService()