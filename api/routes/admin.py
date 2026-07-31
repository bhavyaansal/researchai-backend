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

SAMPLE_SOURCES = [
    {"title": "Climate Change Review 2021", "text": "Global temperatures have risen significantly over the past century due to human activities such as the burning of fossil fuels."},
    {"title": "Climate Change Review 2021", "text": "The Intergovernmental Panel on Climate Change has reported that greenhouse gas emissions are the primary driver of global warming."},
    {"title": "Climate Change Review 2021", "text": "Rising sea levels, extreme weather events, and melting polar ice caps are direct consequences of climate change caused by human activity."},
    {"title": "Machine Learning Foundations", "text": "Neural networks are computational models inspired by the structure and function of biological neural systems in the human brain."},
    {"title": "Machine Learning Foundations", "text": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting model parameters."},
    {"title": "Machine Learning Foundations", "text": "Deep learning models require large amounts of labeled training data to achieve high accuracy on classification tasks."},
    {"title": "Social Media Mental Health Study 2023", "text": "Research indicates a significant correlation between high social media usage and increased levels of anxiety and depression among college students."},
    {"title": "Social Media Mental Health Study 2023", "text": "Studies employing mixed-methods approaches combining surveys and interviews reveal that frequent social media use negatively impacts mental health status."},
    {"title": "Social Media Mental Health Study 2023", "text": "The findings highlight the need for awareness and strategies to mitigate negative mental health outcomes associated with excessive social media use."},
    {"title": "Academic Integrity Systems 2022", "text": "A plagiarism checker functions through a sequential four-phase pipeline that breaks down text structurally and checks for conceptual overlaps."},
    {"title": "Academic Integrity Systems 2022", "text": "Hybrid search combining lexical BM25 matching and semantic vector embeddings provides more accurate plagiarism detection than either method alone."},
]


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