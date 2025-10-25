from pydantic import BaseModel
from typing import List
from app.models.word import EnrichedWord

class BulkEnrichRequest(BaseModel):
    words: List[str]

class BulkEnrichResponse(BaseModel):
    results: List[EnrichedWord]