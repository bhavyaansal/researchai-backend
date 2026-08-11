"""
Simplified semantic search using TF-IDF instead of ChromaDB.
ChromaDB causes silent crashes on Railway free tier (512MB RAM).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db.models import SessionLocal, SourceDocument

_vectorizer = None
_source_texts = []
_source_meta = []


def _build_index():
    global _vectorizer, _source_texts, _source_meta
    db = SessionLocal()
    try:
        sources = db.query(SourceDocument).filter(
            SourceDocument.is_user_upload == 0
        ).all()
        if not sources:
            return
        _source_texts = [s.sentence_text for s in sources]
        _source_meta = [{"title": s.source_title, "text": s.sentence_text} for s in sources]
        _vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
        _vectorizer.fit(_source_texts)
    finally:
        db.close()


def index_source_documents(sources: list[dict]):
    """Called during seeding — just rebuilds the in-memory index."""
    _build_index()


def search_semantic(query_text: str, top_k: int = 5) -> list[dict]:
    global _vectorizer, _source_texts, _source_meta
    if _vectorizer is None:
        _build_index()
    if not _source_texts:
        return []
    try:
        query_vec = _vectorizer.transform([query_text])
        corpus_vecs = _vectorizer.transform(_source_texts)
        scores = cosine_similarity(query_vec, corpus_vecs)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append({
                    "source_title": _source_meta[i]["title"],
                    "source_text": _source_meta[i]["text"],
                    "score": float(scores[i]),
                })
        return results
    except Exception:
        return []


# """
# Semantic search using ChromaDB - free, runs embedded in-process, no server needed.
# This replaces Pinecone/Qdrant in the free stack.
# """
# import chromadb
# from config import settings
# from search.embedder import embed_texts, embed_text

# _client = None
# _collection = None

# COLLECTION_NAME = "source_corpus"


# def get_collection():
#     """Lazily create/load the persistent ChromaDB collection."""
#     global _client, _collection
#     if _client is None:
#         _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
#     if _collection is None:
#         _collection = _client.get_or_create_collection(
#             name=COLLECTION_NAME,
#             metadata={"hnsw:space": "cosine"}
#         )
#     return _collection


# def index_source_documents(sources: list[dict]):
#     """
#     Add source documents to the vector store.
#     sources: [{"id": str, "title": str, "text": str}, ...]
#     """
#     if not sources:
#         return

#     collection = get_collection()
#     texts = [s["text"] for s in sources]
#     raw_embeddings = embed_texts(texts)

#     # Handle both numpy arrays and plain Python lists
#     if hasattr(raw_embeddings, "tolist"):
#         embeddings = raw_embeddings.tolist()
#     else:
#         embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_embeddings]

#     collection.upsert(
#         ids=[s["id"] for s in sources],
#         embeddings=embeddings,
#         documents=texts,
#         metadatas=[{"title": s["title"]} for s in sources],
#     )


# def search_semantic(query_text: str, top_k: int = 5) -> list[dict]:
#     """
#     Search the vector store for semantically similar source sentences.
#     Returns [{"source_title":..., "source_text":..., "score": 0-1}, ...]
#     ChromaDB returns cosine distance (lower = more similar), converted to similarity (1 - dist).
#     """
#     try:
#         collection = get_collection()
#         count = collection.count()
#         if count == 0:
#             return []

#         raw_query = embed_text(query_text)
#         query_embedding = raw_query.tolist() if hasattr(raw_query, "tolist") else list(raw_query)
#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=min(top_k, count),
#         )

#         matches = []
#         docs = results.get("documents", [[]])[0]
#         metas = results.get("metadatas", [[]])[0]
#         distances = results.get("distances", [[]])[0]

#         for doc_text, meta, dist in zip(docs, metas, distances):
#             # In cosine space, cosine_similarity = 1.0 - cosine_distance
#             score = max(0.0, min(1.0, 1.0 - float(dist)))
#             matches.append({
#                 "source_title": meta.get("title", "unknown") if meta else "unknown",
#                 "source_text": doc_text,
#                 "score": score,
#             })

#         return matches
#     except Exception as e:
#         print(f"Error in search_semantic: {e}")
#         return []
