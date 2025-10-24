import os
import httpx
from typing import Dict, Any, Optional

async def fetch_dictionary_data(word: str) -> Optional[Dict[str, Any]]:
    """Gọi Free Dictionary API để lấy dữ liệu thô."""
    base_url = os.getenv("DICTIONARY_API_URL")
    url = f"{base_url}{word}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return data[0] if isinstance(data, list) and data else None
            return None
        except Exception as e:
            print(f"Dictionary API error: {e}")
            return None