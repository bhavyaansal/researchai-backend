"""
Admin routes — for seeding corpus on Railway deployment.
POST /admin/seed — seeds the source corpus into SQLite + ChromaDB.
Protected by ADMIN_SECRET env variable.
"""
import os
from fastapi import APIRouter, Header, HTTPException
from db.models import SessionLocal, init_db, SourceDocument
from search.semantic import index_source_documents

router = APIRouter(prefix="/admin", tags=["Admin"])

SAMPLE_SOURCES = []


@router.post("/seed")
def seed_corpus(x_admin_secret: str = Header(...)):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    if not admin_secret or x_admin_secret != admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    init_db()
    db = SessionLocal()
    chroma_records = []

    try:
        for src in SAMPLE_SOURCES:
            existing = db.query(SourceDocument).filter(
                SourceDocument.sentence_text == src["text"]
            ).first()
            if existing:
                continue
            doc = SourceDocument(source_title=src["title"], sentence_text=src["text"])
            db.add(doc)
            db.flush()
            chroma_records.append({"id": doc.id, "title": src["title"], "text": src["text"]})

        db.commit()
        if chroma_records:
            index_source_documents(chroma_records)

        return {
            "status": "success",
            "seeded": len(chroma_records),
            "message": f"Seeded {len(chroma_records)} new documents"
        }
    finally:
        db.close()


@router.get("/health")
def health():
    db = SessionLocal()
    try:
        count = db.query(SourceDocument).count()
        return {"corpus_size": count, "status": "ok"}
    finally:
        db.close()