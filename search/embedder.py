"""
Embedding model wrapper using sentence-transformers.
Loaded once and cached (loading the model takes a few seconds, so don't reload per call).
"""
from sentence_transformers import SentenceTransformer
from config import settings

_model = None


def get_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]):
    """Encode a list of strings into embedding vectors. Returns a numpy array."""
    model = get_model()
    return model.encode(texts, show_progress_bar=False)


def embed_text(text: str):
    """Encode a single string into one embedding vector."""
    return embed_texts([text])[0]
