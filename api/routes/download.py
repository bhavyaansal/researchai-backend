from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
import io
import os
import datetime

from db.models import get_db, Job, FlaggedSpan, User
from auth.dependencies import get_current_user
from config import settings
from .ieee_pdf import build_ieee_pdf, reconstruct_document_runs
from .analysis_pdf import build_report_pdf, job_to_report

router = APIRouter(prefix="/download", tags=["Download"])


def reconstruct_document(full_text: str, spans: list) -> str:
    if not full_text:
        return ""
    sorted_spans = sorted(spans, key=lambda s: s.start_char)
    result = []
    last_idx = 0
    for span in sorted_spans:
        if span.start_char > last_idx:
            result.append(full_text[last_idx:span.start_char])
        val = span.rewritten_text if span.rewritten_text is not None else span.original_text
        result.append(val)
        last_idx = span.end_char
    if last_idx < len(full_text):
        result.append(full_text[last_idx:])
    return "".join(result)


@router.get("/report/{job_id}")
def download_report(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the plagiarism analysis as a PDF: integrity summary, top sources,
    highlighted document text and a segment-by-segment breakdown.
    """
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spans = db.query(FlaggedSpan).filter(FlaggedSpan.job_id == job_id).all()

    pdf_bytes = build_report_pdf(job_to_report(job, spans))

    base, _ext = os.path.splitext((job.filename or "document").replace(" ", "_"))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={base}_plagiarism_report.pdf",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.get("/rewritten/{job_id}")
def download_rewritten(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the merged document — original text plus AI-rewritten spans —
    as a single downloadable PDF, laid out like an IEEE conference paper
    (title block + two-column body). Rewritten portions are shown in
    blue italics so the reader can see exactly what changed.
    """
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spans = db.query(FlaggedSpan).filter(FlaggedSpan.job_id == job_id).all()

    runs = reconstruct_document_runs(job.full_text, spans)
    if not runs:
        runs = [("No content available. The original document was empty.", False)]

    score = (job.global_similarity_score or 0.0) * 100
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Title: use the filename (minus extension) as a stand-in paper title.
    base_name, ext = os.path.splitext(job.filename or "Untitled Document")
    title = base_name.replace("_", " ").replace("-", " ").strip() or "Untitled Document"

    # Same path pipeline_task.py used when parsing — where any
    # extracted figures/tables for this job were saved (DOCX only,
    # for now — see ingestion/docx_parser.py).
    assets_dir = os.path.join(settings.UPLOAD_DIR, str(job.id), "assets")

    pdf_bytes = build_ieee_pdf(
        title=title,
        filename=job.filename or "Untitled Document",
        job_id=str(job.id),
        similarity_pct=score,
        generated_at=generated_at,
        runs=runs,
        assets_dir=assets_dir,
    )

    safe_filename = (job.filename or "document").replace(" ", "_")
    base, _ext = os.path.splitext(safe_filename)
    download_name = f"{base}_rewritten_ieee.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={download_name}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )
