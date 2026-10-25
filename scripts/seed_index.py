"""
Seed script: populates both the SQLite database (SourceDocument table) and
ChromaDB vector store with a small sample corpus so you can test the pipeline
before connecting a real source corpus.

Run from inside the backend folder:
    python ../scripts/seed_index.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.models import SessionLocal, init_db, SourceDocument
from search.semantic import index_source_documents

def main():
    init_db()
    db = SessionLocal()

    db.commit()

    # Always ensure ChromaDB is synced with all SourceDocuments in SQLite
    all_docs = db.query(SourceDocument).all()
    chroma_records = [{"id": str(doc.id), "title": doc.source_title, "text": doc.sentence_text} for doc in all_docs]

    db.close()

    if chroma_records:
        index_source_documents(chroma_records)
        print(f"Synced {len(chroma_records)} source documents into SQLite + ChromaDB.")

if __name__ == "__main__":
    main()