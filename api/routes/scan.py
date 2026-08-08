"""
GET /scan/{job_id} - returns the current status of a job (for polling).
Only returns jobs owned by the authenticated user.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.models import get_db, Job, User
from db.schemas import JobResponse
from auth.dependencies import get_current_user

router = APIRouter()


import threading
from tasks.pipeline_task import run_pipeline


@router.get("/scan/{job_id}", response_model=JobResponse)
def get_scan_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/rewrite/{job_id}", response_model=JobResponse)
def retrigger_rewrite(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("scanning", "rewriting", "validating"):
        job.status = "queued"
        db.commit()
        db.refresh(job)
        threading.Thread(target=run_pipeline, args=(job.id,), daemon=True).start()
    return job

