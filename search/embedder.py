"""
Lightweight embedding model wrapper.
Uses fastembed (ONNX runtime) — 50MB RAM, fast CPU inference, no heavy PyTorch needed.
Fallback to sentence-transformers if fastembed is not installed.
"""
import numpy as np

_model = None


def get_model():
    """Lazily load and cache fastembed (ONNX) or sentence-transformers."""
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
            _model = ("fastembed", TextEmbedding(model_name="BAAI/bge-small-en-v1.5"))
            print("Loaded fastembed (ONNX runtime) embedding model successfully.")
        except Exception as e:
            print(f"fastembed not available ({e}), trying sentence-transformers...")
            try:
                from sentence_transformers import SentenceTransformer
                from config import settings
                _model = ("sentence-transformers", SentenceTransformer(settings.EMBEDDING_MODEL))
                print("Loaded sentence-transformers model successfully.")
            except Exception as e2:
                print(f"sentence-transformers not available ({e2}), using tfidf fallback...")
                _model = ("tfidf", None)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of strings into embedding vectors. Returns a numpy array."""
    if not texts:
        return np.array([])

    model_type, model = get_model()

    if model_type == "fastembed":
        try:
            embeddings = list(model.embed(texts))
            return np.array(embeddings)
        except Exception as e:
            print(f"Error in fastembed encoding: {e}")
            return np.zeros((len(texts), 384))

    elif model_type == "sentence-transformers":
        try:
            return model.encode(texts, show_progress_bar=False)
        except Exception as e:
            print(f"Error in sentence-transformers encoding: {e}")
            return np.zeros((len(texts), 384))

    else:
        # TF-IDF + Character n-gram fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
            matrix = vec.fit_transform(texts).toarray()
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return matrix / norms
        except Exception as e:
            print(f"Error in tfidf fallback: {e}")
            return np.zeros((len(texts), 384))


def embed_text(text: str) -> np.ndarray:
    """Encode a single string into one embedding vector."""
    return embed_texts([text])[0]
