"""
GET  /report/{job_id}                    - full JSON report
GET  /jobs                               - list all jobs for current user
GET  /report/{job_id}/download/report    - plagiarism analysis as plain text
GET  /report/{job_id}/download/rewritten - polished (rewritten) document as plain text
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from typing import List
import datetime

from db.models import get_db, Job, FlaggedSpan, User
from db.schemas import ReportResponse, JobResponse
from auth.dependencies import get_current_user

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
    return ReportResponse(job=job, spans=spans, full_text=job.full_text)


@router.get("/jobs", response_model=List[JobResponse])
def list_my_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    jobs = (
        db.query(Job)
        .filter(Job.user_id == current_user.id)
        .order_by(Job.created_at.desc())
        .all()
    )
    return jobs


@router.get("/report/{job_id}/download/report", response_class=PlainTextResponse)
def download_plagiarism_report(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a plain-text plagiarism analysis report for the job.
    Flutter downloads this and saves it via DownloadHelper.
    """
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spans = db.query(FlaggedSpan).filter(FlaggedSpan.job_id == job_id).all()
    score = (job.global_similarity_score or 0) * 100

    lines = []
    lines.append("RESEARCHAI PLAGIARISM ANALYSIS REPORT")
    lines.append("=" * 50)
    lines.append(f"File       : {job.filename}")
    lines.append(f"Scanned on : {job.created_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Status     : {job.status}")
    lines.append(f"Overall Similarity: {score:.1f}%")
    lines.append("")

    # Deduplicate sources
    source_scores: dict[str, float] = {}
    for span in spans:
        title = span.matched_source_title or "Unknown"
        if span.combined_score > source_scores.get(title, 0):
            source_scores[title] = span.combined_score

    lines.append("SOURCE BREAKDOWN")
    lines.append("-" * 50)
    for title, sc in sorted(source_scores.items(), key=lambda x: -x[1]):
        lines.append(f"  {title}: {sc * 100:.0f}%")
    lines.append("")

    lines.append(f"FLAGGED SPANS ({len(spans)} total)")
    lines.append("-" * 50)
    for i, span in enumerate(spans, 1):
        lines.append(f"\nSpan {i}:")
        lines.append(f"  Source     : {span.matched_source_title or 'Unknown'}")
        lines.append(f"  Similarity : {span.combined_score * 100:.1f}%")
        lines.append(f"  Resolved   : {'Yes' if span.resolved else 'No'}")
        lines.append(f"  Original   :")
        lines.append(f"    {span.original_text}")
        if span.rewritten_text:
            lines.append(f"  Rewritten  :")
            lines.append(f"    {span.rewritten_text}")
        lines.append("-" * 50)

    return "\n".join(lines)


@router.get("/report/{job_id}/download/rewritten", response_class=PlainTextResponse)
def download_rewritten_document(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the polished (rewritten) document as plain text.
    Uses rewritten_text where available, falls back to original.
    """
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spans = db.query(FlaggedSpan).filter(FlaggedSpan.job_id == job_id).all()
    score = (job.global_similarity_score or 0) * 100

    lines = []
    lines.append("RESEARCHAI — POLISHED DOCUMENT")
    lines.append("=" * 50)
    lines.append(f"Original file : {job.filename}")
    lines.append(f"Generated on  : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append(f"Final Similarity: {score:.1f}%")
    lines.append("")
    lines.append("REWRITTEN CONTENT")
    lines.append("=" * 50)
    lines.append("")

    if not spans:
        lines.append("No flagged content — document was already original.")
    else:
        for i, span in enumerate(spans, 1):
            original_score = span.combined_score * 100
            final_score = (span.final_score_after_rewrite or span.combined_score) * 100
            lines.append(
                f"[Segment {i}] {original_score:.0f}% → {final_score:.0f}% similarity"
                f" | Source: {span.matched_source_title or 'Unknown'}"
            )
            lines.append("")
            lines.append(span.rewritten_text or span.original_text)
            lines.append("")
            lines.append("-" * 50)
            lines.append("")

    return "\n".join(lines)
