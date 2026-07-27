"""
FastAPI entry point. Wires up routes, CORS, and database initialization.

Run locally with:
    uvicorn main:app --reload

Run via Docker:
    docker compose up backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.models import init_db
from api.routes import uploads, scan, report, download
from auth.routes import router as auth_router

app = FastAPI(
    title="Plagiarism Detection & Rewriting API",
    description="Free, self-hosted backend: parsing, hybrid search, and LLM rewriting",
    version="1.0.0",
)

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def health_check():
    return {"status": "ok", "service": "plagiarism-backend"}


# Mount routers
app.include_router(auth_router)
app.include_router(uploads.router, tags=["Upload"])
app.include_router(scan.router, tags=["Scan"])
app.include_router(report.router, tags=["Report"])
app.include_router(download.router, tags=["Download"])
