from pydantic import BaseModel, model_validator
from typing import List, Optional, Dict, Any, Union

class PhoneticDetail(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None

class Phonetics(BaseModel):
    uk: Optional[PhoneticDetail] = None
    us: Optional[PhoneticDetail] = None

class ExamplePair(BaseModel):
    en: Optional[str] = None
    vi: Optional[str] = None

class TranslatedPhrase(BaseModel):
    en: str
    vi: str

class EnrichedWord(BaseModel):
    word: str
    word_vn: Optional[str] = None
    phonetics: Optional[Union[Phonetics, List[Dict[str, Any]], Dict[str, str]]] = None
    partOfSpeech: Optional[str] = None
    definition_en: Optional[str] = None
    definition_vi: Optional[str] = None
    examples: List[ExamplePair] = []
    idioms_collocations: List[TranslatedPhrase] = []
    phrasal_verbs: List[TranslatedPhrase] = []
    synonyms: List[str] = []
    source: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def clean_and_transform_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            phonetics_value = data.get('phonetics')
            if isinstance(phonetics_value, dict) and not ('uk' in phonetics_value and isinstance(phonetics_value.get('uk'), dict)):
                new_phonetics = Phonetics(
                    uk=PhoneticDetail(text=phonetics_value.get('uk')),
                    us=PhoneticDetail(text=phonetics_value.get('us'))
                )
                data['phonetics'] = new_phonetics.model_dump(exclude_none=True)
            elif isinstance(phonetics_value, list):
                new_phonetics = Phonetics()
                if len(phonetics_value) > 0: new_phonetics.uk = PhoneticDetail(**phonetics_value[0])
                if len(phonetics_value) > 1: new_phonetics.us = PhoneticDetail(**phonetics_value[1])
                data['phonetics'] = new_phonetics.model_dump(exclude_none=True)
        return data