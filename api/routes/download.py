from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
import io
import os
import datetime

from db.models import get_db, Job, FlaggedSpan, User
from auth.dependencies import get_current_user
from .ieee_pdf import build_ieee_pdf, reconstruct_document_runs

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
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spans = db.query(FlaggedSpan).filter(FlaggedSpan.job_id == job_id).all()

    # Generate a beautiful textual report
    score = (job.global_similarity_score or 0.0) * 100

    buf = io.StringIO()
    buf.write("═════════════════════════════════════════════════════════════════\n")
    buf.write("                      RESEARCHAI REPORT ANALYSIS                 \n")
    buf.write("═════════════════════════════════════════════════════════════════\n")
    buf.write(f"Document File: {job.filename}\n")
    buf.write(f"Job Reference ID: {job.id}\n")
    buf.write(f"Scan Completed: {job.updated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    buf.write(f"Overall Similarity Score: {score:.1f}%\n")
    buf.write("═════════════════════════════════════════════════════════════════\n\n")

    # Source breakdown
    sources = {}
    for span in spans:
        title = span.matched_source_title or "Unknown Source"
        sources[title] = max(sources.get(title, 0.0), span.combined_score * 100)

    buf.write(f"SOURCE BREAKDOWN ({len(sources)} unique source(s)):\n")
    buf.write("─" * 65 + "\n")
    if not sources:
        buf.write("No matching plagiarism sources detected. Content is original.\n")
    else:
        for title, pct in sorted(sources.items(), key=lambda item: item[1], reverse=True):
            buf.write(f"• {title}: {pct:.0f}% Similarity\n")
    buf.write("\n")

    # Flagged segments details
    buf.write(f"FLAGGED SEGMENTS DETAILS ({len(spans)} segments):\n")
    buf.write("─" * 65 + "\n")
    for i, span in enumerate(spans, 1):
        status = "Resolved (Rewritten)" if span.resolved == 1 else "Flagged"
        buf.write(f"[{i}] Status: {status}\n")
        buf.write(f"    Source: {span.matched_source_title or 'Unknown'}\n")
        buf.write(f"    Original Similarity: {span.combined_score * 100:.0f}%\n")
        if span.final_score_after_rewrite is not None:
            buf.write(f"    Final Similarity after rewrite: {span.final_score_after_rewrite * 100:.0f}%\n")
        buf.write(f"    Original Text:\n")
        buf.write(f"      \"{span.original_text}\"\n")
        if span.rewritten_text:
            buf.write(f"    AI Suggested Rewrite:\n")
            buf.write(f"      \"{span.rewritten_text}\"\n")
        buf.write("\n")

    report_content = buf.getvalue()
    buf.close()

    # Return response
    safe_filename = job.filename.replace(" ", "_")
    return Response(
        content=report_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename={safe_filename}_plagiarism_report.txt",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
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

    pdf_bytes = build_ieee_pdf(
        title=title,
        filename=job.filename or "Untitled Document",
        job_id=str(job.id),
        similarity_pct=score,
        generated_at=generated_at,
        runs=runs,
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