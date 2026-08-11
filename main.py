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

SAMPLE_SOURCES = [
    {"title": "Climate Change Review 2021", "text": "Global temperatures have risen significantly over the past century due to human activities such as the burning of fossil fuels."},
    {"title": "Climate Change Review 2021", "text": "The Intergovernmental Panel on Climate Change has reported that greenhouse gas emissions are the primary driver of global warming."},
    {"title": "Machine Learning Foundations", "text": "Neural networks are computational models inspired by the structure and function of biological neural systems in the human brain."},
    {"title": "Machine Learning Foundations", "text": "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting model parameters."},
    {"title": "Machine Learning Foundations", "text": "Deep learning models require large amounts of labeled training data to achieve high accuracy on classification tasks."},
    {"title": "Social Media Mental Health Study 2023", "text": "Research indicates a significant correlation between high social media usage and increased levels of anxiety and depression among college students."},
    {"title": "Social Media Mental Health Study 2023", "text": "Studies employing mixed-methods approaches combining surveys and interviews reveal that frequent social media use negatively impacts mental health status."},
    {"title": "Academic Integrity Systems 2022", "text": "A plagiarism checker functions through a sequential four-phase pipeline that breaks down text structurally and checks for conceptual overlaps."},
]


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


