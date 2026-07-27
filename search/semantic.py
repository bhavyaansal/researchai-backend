"""
Semantic search using ChromaDB - free, runs embedded in-process, no server needed.
This replaces Pinecone/Qdrant in the free stack.
"""
import chromadb
from config import settings
from search.embedder import embed_texts, embed_text

_client = None
_collection = None

COLLECTION_NAME = "source_corpus"


def get_collection():
    """Lazily create/load the persistent ChromaDB collection."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    if _collection is None:
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def index_source_documents(sources: list[dict]):
    """
    Add source documents to the vector store.
    sources: [{"id": str, "title": str, "text": str}, ...]
    """
    if not sources:
        return

    collection = get_collection()
    texts = [s["text"] for s in sources]
    embeddings = embed_texts(texts).tolist()

    collection.upsert(
        ids=[s["id"] for s in sources],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"title": s["title"]} for s in sources],
    )


def search_semantic(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Search the vector store for semantically similar source sentences.
    Returns [{"source_title":..., "source_text":..., "score": 0-1}, ...]
    ChromaDB returns distance (lower = more similar), so we convert to a similarity score.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_text(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    matches = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc_text, meta, dist in zip(docs, metas, distances):
        # Convert cosine distance to similarity: similarity = 1 - distance (clamped)
        score = max(0.0, min(1.0, 1.0 - dist))
        matches.append({
            "source_title": meta.get("title", "unknown"),
            "source_text": doc_text,
            "score": score,
        })

    return matches
