"""
Lightweight text embedder and semantic similarity calculator.
Uses character n-gram TF-IDF vectors for fast in-process similarity computation without external model downloads.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute character n-gram TF-IDF cosine similarity between two texts."""
    if not text_a or not text_b:
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        matrix = vec.fit_transform([text_a, text_b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return max(0.0, min(1.0, float(sim)))
    except Exception:
        return 0.0


def embed_text(text: str) -> np.ndarray:
    """Encode a single string into a TF-IDF vector."""
    if not text:
        return np.zeros(100)
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        return vec.fit_transform([text]).toarray()[0]
    except Exception:
        return np.zeros(100)


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Encode a list of strings into TF-IDF vectors."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        matrix = vec.fit_transform(texts).toarray()
        return [row for row in matrix]
    except Exception:
        return [np.zeros(100) for _ in texts]


# """
# Lightweight embedder using scikit-learn TF-IDF vectors.
# No model download needed — works instantly on Railway free tier.
# Replaces sentence-transformers to avoid the 90MB model download timeout.
# """
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# _vectorizer = None
# _corpus_texts = []


# def _get_vectorizer():
#     global _vectorizer
#     if _vectorizer is None:
#         _vectorizer = TfidfVectorizer(
#             max_features=5000,
#             ngram_range=(1, 2),
#             stop_words='english',
#         )
#     return _vectorizer


# def embed_texts(texts: list[str]) -> np.ndarray:
#     global _corpus_texts
#     vectorizer = _get_vectorizer()
#     combined = list(set(_corpus_texts + list(texts)))
#     _corpus_texts = combined
#     vectorizer.fit(combined)
#     matrix = vectorizer.transform(texts)
#     return matrix.toarray()


# def embed_text(text: str) -> np.ndarray:
#     return embed_texts([text])[0]

# --------------------

# """
# Lightweight embedding model wrapper.
# Uses fastembed (ONNX runtime) — 50MB RAM, fast CPU inference, no heavy PyTorch needed.
# Fallback to sentence-transformers if fastembed is not installed.
# """
# import numpy as np

# _model = None


# def get_model():
#     """Lazily load and cache fastembed (ONNX) or sentence-transformers."""
#     global _model
#     if _model is None:
#         try:
#             from fastembed import TextEmbedding
#             _model = ("fastembed", TextEmbedding(model_name="BAAI/bge-small-en-v1.5"))
#             print("Loaded fastembed (ONNX runtime) embedding model successfully.")
#         except Exception as e:
#             print(f"fastembed not available ({e}), trying sentence-transformers...")
#             try:
#                 from sentence_transformers import SentenceTransformer
#                 from config import settings
#                 _model = ("sentence-transformers", SentenceTransformer(settings.EMBEDDING_MODEL))
#                 print("Loaded sentence-transformers model successfully.")
#             except Exception as e2:
#                 print(f"sentence-transformers not available ({e2}), using tfidf fallback...")
#                 _model = ("tfidf", None)
#     return _model


# def embed_texts(texts: list[str]) -> np.ndarray:
#     """Encode a list of strings into embedding vectors. Returns a numpy array."""
#     if not texts:
#         return np.array([])

#     model_type, model = get_model()

#     if model_type == "fastembed":
#         try:
#             embeddings = list(model.embed(texts))
#             return np.array(embeddings)
#         except Exception as e:
#             print(f"Error in fastembed encoding: {e}")
#             return np.zeros((len(texts), 384))

#     elif model_type == "sentence-transformers":
#         try:
#             return model.encode(texts, show_progress_bar=False)
#         except Exception as e:
#             print(f"Error in sentence-transformers encoding: {e}")
#             return np.zeros((len(texts), 384))

#     else:
#         # TF-IDF + Character n-gram fallback
#         try:
#             from sklearn.feature_extraction.text import TfidfVectorizer
#             vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
#             matrix = vec.fit_transform(texts).toarray()
#             norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#             norms[norms == 0] = 1.0
#             return matrix / norms
#         except Exception as e:
#             print(f"Error in tfidf fallback: {e}")
#             return np.zeros((len(texts), 384))


# def embed_text(text: str) -> np.ndarray:
#     """Encode a single string into one embedding vector."""
#     return embed_texts([text])[0]
