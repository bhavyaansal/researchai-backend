"""
FastAPI entry point. Wires up routes, CORS, database initialization,
and auto-seeds the corpus on first startup if empty.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from db.models import init_db, SessionLocal, SourceDocument
from api.routes import upload, scan, report
from api.routes.admin import router as admin_router, SAMPLE_SOURCES
from auth.routes import router as auth_router
from search.semantic import index_source_documents

app = FastAPI(
    title="Plagiarism Detection & Rewriting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Initialize DB tables
    init_db()

    # Auto-seed corpus if empty
    db: Session = SessionLocal()
    try:
        count = db.query(SourceDocument).count()
        if count == 0:
            print("Corpus is empty — seeding source documents...")
            chroma_records = []
            for src in SAMPLE_SOURCES:
                doc = SourceDocument(
                    source_title=src["title"],
                    sentence_text=src["text"]
                )
                db.add(doc)
                db.flush()
                chroma_records.append({
                    "id": doc.id,
                    "title": src["title"],
                    "text": src["text"]
                })
            db.commit()
            if chroma_records:
                index_source_documents(chroma_records)
            print(f"Auto-seeded {len(chroma_records)} source documents.")
        else:
            print(f"Corpus already has {count} documents — skipping seed.")
    except Exception as e:
        print(f"Seeding error (non-fatal): {e}")
    finally:
        db.close()


@app.get("/")
def health_check():
    return {"status": "ok", "service": "plagiarism-backend"}


# Mount routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(upload.router, tags=["Upload"])
app.include_router(scan.router, tags=["Scan"])
app.include_router(report.router, tags=["Report"])
