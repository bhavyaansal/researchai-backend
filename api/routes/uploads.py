"""
POST /upload - accepts a PDF or DOCX file, saves it, creates a Job row owned
by the authenticated user, and kicks off the pipeline in a background thread.
"""
import os
import uuid
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import get_db, Job, User
from db.schemas import JobResponse
from storage.file_store import save_upload
from tasks.pipeline_task import run_pipeline
from auth.dependencies import get_current_user
from config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class TextUploadRequest(BaseModel):
    text: str


@router.post("/upload/text", response_model=JobResponse)
async def upload_text(
    payload: TextUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import uuid, os
    job_id = str(uuid.uuid4())
    # Save text as a temporary .txt file so the pipeline can process it
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(payload.text)

    job = Job(
        id=job_id,
        user_id=current_user.id,
        filename="text_input.txt",
        file_path=file_path,
        status="uploaded",
    )
    db.add(job)
    db.commit()
    db.refresh(job)


    import threading
    from tasks.pipeline_task import run_pipeline
    threading.Thread(target=run_pipeline, args=(job.id,), daemon=True).start()
    return job


@router.post("/upload", response_model=JobResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf, .docx and .txt files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job_id, saved_path = save_upload(file_bytes, file.filename)

    job = Job(
        id=job_id,
        user_id=current_user.id,
        filename=file.filename,
        file_path=saved_path,
        status="uploaded",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Kick off the pipeline in the background so the request returns immediately
    thread = threading.Thread(target=run_pipeline, args=(job.id,), daemon=True)
    thread.start()

    return job
