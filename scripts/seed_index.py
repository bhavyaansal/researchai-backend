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

SAMPLE_SOURCES = [
    {
        "title": "Climate Change Review 2021",
        "text": "Global temperatures have risen significantly over the past century due to human activities such as the burning of fossil fuels.",
    },
    {
        "title": "Climate Change Review 2021",
        "text": "The Intergovernmental Panel on Climate Change has reported that greenhouse gas emissions are the primary driver of global warming.",
    },
    {
        "title": "Social Media & Adolescent Mental Health (2020)",
        "text": "Investigate the relationship between social media usage and mental health issues, including anxiety and depression, among college students.",
    },
    {
        "title": "Social Media & Adolescent Mental Health (2020)",
        "text": "A review of existing literature reveals mixed outcomes regarding social media's impact on mental health. Some studies suggest that social media can enhance social support and well-being, while others point to increased risks of anxiety, depression, and loneliness.",
    },
    {
        "title": "Social Media & Adolescent Mental Health (2020)",
        "text": "The study used a mixed-methods design, combining quantitative surveys with qualitative interviews to gather insights into the relationships between social media use patterns and mental health status among randomly selected college students.",
    },
    {
        "title": "Machine Learning Foundations",
        "text": "Neural networks are computational models inspired by the structure and function of biological neural systems in the brain.",
    },
    {
        "title": "Machine Learning Foundations",
        "text": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting model parameters.",
    },
    {
        "title": "Quantum Computing Basics",
        "text": "Quantum computers use qubits, which can exist in superposition, allowing them to process multiple states simultaneously.",
    },
]


def main():
    init_db()
    db = SessionLocal()

    for src in SAMPLE_SOURCES:
        existing = db.query(SourceDocument).filter(
            SourceDocument.sentence_text == src["text"]
        ).first()
        if not existing:
            doc = SourceDocument(source_title=src["title"], sentence_text=src["text"])
            db.add(doc)

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
