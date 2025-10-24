import os
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load model một lần và cache lại."""
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
    print(f"Loading embedding model: {model_name}...")
    return SentenceTransformer(model_name)

def embed_text(text: str) -> list[float]:
    """Tạo vector embedding cho một chuỗi văn bản."""
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()