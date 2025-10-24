from pydantic import BaseModel
from typing import List, Optional

class PhoneticDetail(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None

class Phonetics(BaseModel):
    uk: Optional[PhoneticDetail] = None
    us: Optional[PhoneticDetail] = None

class ExamplePair(BaseModel):
    en: Optional[str] = None
    vi: Optional[str] = None

class EnrichedWord(BaseModel):
    word: str
    phonetics: Optional[Phonetics] = None
    partOfSpeech: Optional[str] = None
    definition_en: Optional[str] = None
    definition_vi: Optional[str] = None
    examples: List[ExamplePair] = []
    idioms: List[str] = []
    source: Optional[str] = None