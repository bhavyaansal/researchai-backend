from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.models import init_db, SessionLocal, SourceDocument
from api.routes import uploads as upload, scan, report, download, paraphrase
from api.routes.admin import router as admin_router
from auth.routes import router as auth_router
from search.semantic import index_source_documents

app = FastAPI(title="Plagiarism Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_SOURCES = []

@app.on_event("startup")
def on_startup():
    try:
        init_db()
        db = SessionLocal()
        try:
            count = db.query(SourceDocument).count()
            if count == 0:
                for src in SAMPLE_SOURCES:
                    doc = SourceDocument(
                        source_title=src["title"],
                        sentence_text=src["text"]
                    )
                    db.add(doc)
                db.commit()
                print(f"Seeded {len(SAMPLE_SOURCES)} source documents.")
            else:
                print(f"Corpus has {count} documents.")
        finally:
            db.close()
        # Refresh semantic vector index
        index_source_documents([])
    except Exception as e:
        print(f"Startup error (non-fatal): {e}")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "plagiarism-backend"}


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(upload.router, tags=["Upload"])
app.include_router(scan.router, tags=["Scan"])
app.include_router(report.router, tags=["Report"])
app.include_router(download.router)
app.include_router(paraphrase.router, tags=["Paraphrase"])


