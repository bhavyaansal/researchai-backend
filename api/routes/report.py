"""
GET  /report/{job_id}                    - full JSON report
GET  /jobs                               - list all jobs for current user
GET  /report/{job_id}/download/report    - plagiarism analysis PDF   (same as /download/report/{job_id})
GET  /report/{job_id}/download/rewritten - polished document PDF     (same as /download/rewritten/{job_id})
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.models import get_db, Job, FlaggedSpan, User
from db.schemas import ReportResponse, JobResponse, FlaggedSpanResponse
from auth.dependencies import get_current_user
from .download import download_report, download_rewritten

router = APIRouter()


@router.get("/report/{job_id}", response_model=ReportResponse)
def get_report(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    spans = db.query(FlaggedSpan).filter(FlaggedSpan.job_id == job_id).all()

    return ReportResponse(
        job_id=str(job.id),
        filename=job.filename or "Untitled document",
        status=job.status or "done",
        global_similarity_score=job.global_similarity_score or 0.0,
        flagged_spans=[FlaggedSpanResponse.from_db(s) for s in spans],
        full_text=job.full_text,
        rewritten_full_text=None,
        created_at=job.created_at,
    )


@router.get("/jobs", response_model=List[JobResponse])
def list_my_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Job)
        .filter(Job.user_id == current_user.id)
        .order_by(Job.created_at.desc())
        .all()
    )


# Legacy URLs: they used to return plain text. They now return the same PDFs
# as the /download/* routes so no client can get the old .txt output.
@router.get("/report/{job_id}/download/report")
def legacy_download_report(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return download_report(job_id, db, current_user)


@router.get("/report/{job_id}/download/rewritten")
def legacy_download_rewritten(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return download_rewritten(job_id, db, current_user)