"""
Pipeline orchestrator — conservative Gemini usage.
Never hangs: every Gemini call should have a timeout inside rewrite_paragraph().
On any failure, keeps original text and continues.
"""

import time
import os

from db.models import (
    SessionLocal,
    Job,
    FlaggedSpan,
)
from ingestion.parser import extract_text, split_into_paragraphs
from search.lexical import build_bm25_index
from search.hybrid import hybrid_search
from mapping.coordinate_mapper import map_span_to_match
from rewriter.rewrite_chain import rewrite_paragraph
from rewriter.validator import validate_rewrite
from scoring.similarity import global_document_score
from config import settings

# Max number of live web-search fallback attempts allowed per document
# scan. Each attempt can take up to a few seconds (network round trip to
# the self-hosted SearXNG instance), so this caps how much latency one
# scan can add.
#
# Raised from 8 -> 25 now that the local SourceDocument corpus is
# intentionally empty (see main.py) — every flagged-length paragraph
# falls through to web search, so a low cap would leave most of a
# normal-length document unchecked past the first few paragraphs.
# Worst case adds ~25 * 6s (SearXNG timeout) of latency to a scan;
# tune this down if scans start feeling too slow on your free-tier
# hosting, or up if documents are still running out of budget.
MAX_WEB_SEARCHES_PER_JOB = 25


class WebBudget:
    """Tracks how many web-search fallback attempts a single scan has left."""

    def __init__(self, limit: int):
        self.remaining = limit

    def use(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            return True
        return False


def run_pipeline(job_id: str):
    db = SessionLocal()

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        # --------------------------------------------------
        # Stage 1 : Parsing
        # --------------------------------------------------
        job.status = "parsing"
        db.commit()

        # Extracted figures/tables (currently DOCX only — see
        # ingestion/docx_parser.py) are saved here, so the download
        # endpoint can find them again later via the same path.
        assets_dir = os.path.join(settings.UPLOAD_DIR, job.id, "assets")

        full_text = extract_text(job.file_path, assets_dir=assets_dir)
        job.full_text = full_text

        paragraphs = split_into_paragraphs(full_text)

        # --------------------------------------------------
        # Stage 2 : Scanning
        # --------------------------------------------------
        job.status = "scanning"
        db.commit()

        bm25_index, sources = build_bm25_index(db)

        flagged = []
        web_budget = WebBudget(MAX_WEB_SEARCHES_PER_JOB)

        for para in paragraphs:
            text = para["text"].strip()

            # Ignore very small paragraphs
            if len(text) < 30:
                continue

            match = hybrid_search(
                text,
                bm25_index,
                sources,
                top_k=5,
                web_budget=web_budget,
            )

            if (
                match
                and match["combined_score"]
                >= settings.SIMILARITY_FLAG_THRESHOLD
            ):
                flagged.append((para, match))

        # NOTE: previously capped to flagged[:5] to limit Gemini usage.
        # Now rewriting every flagged paragraph — the 6s sleep between
        # calls below is what keeps this inside the Gemini free-tier
        # RPM limit, so removing the cap is safe, just slower on
        # documents with many flagged spans.

        # --------------------------------------------------
        # Stage 3 : Coordinate Mapping
        # --------------------------------------------------
        job.status = "mapping"
        db.commit()

        span_records = []

        for para, match in flagged:
            record = map_span_to_match(para, match)

            db_span = FlaggedSpan(
                job_id=job.id,
                **record,
            )

            db.add(db_span)
            span_records.append(db_span)

        db.commit()

        # --------------------------------------------------
        # Stage 4–6 : Rewriting
        # --------------------------------------------------
        job.status = "rewriting"
        db.commit()

        for i, span in enumerate(span_records):

            print(
                f"[{i+1}/{len(span_records)}] "
                f"Rewriting span | score={span.combined_score:.2f}"
                + (f" | web:{span.source_url}" if span.source_url else "")
            )

            # Already below target -> don't waste Gemini call
            if span.combined_score <= settings.TARGET_GLOBAL_THRESHOLD:
                span.rewritten_text = span.original_text
                span.rewrite_attempts = 0
                span.final_score_after_rewrite = span.combined_score
                span.resolved = 1
                db.commit()
                continue

            try:
                rewritten = rewrite_paragraph(
                    span.original_text,
                    span.combined_score,
                )

                scores = validate_rewrite(
                    rewritten,
                    span.matched_source_text,
                )

                span.rewritten_text = rewritten
                span.final_score_after_rewrite = scores["combined_score"]

            except RuntimeError as e:
                print(f"Rewrite skipped: {e}")

                # Keep original if Gemini fails
                span.rewritten_text = span.original_text
                span.final_score_after_rewrite = span.combined_score

            span.rewrite_attempts = 1
            span.resolved = 1
            db.commit()

            # Stay inside Gemini free-tier RPM
            if i < len(span_records) - 1:
                time.sleep(6)

        # --------------------------------------------------
        # NOTE: Auto-ingest of uploaded docs into corpus is intentionally DISABLED.
        # Auto-ingesting user uploads causes "self-matching" on subsequent scans:
        # every paragraph would match its own previously uploaded version at 100%.
        # The reference corpus should only contain manually curated academic sources.
        # (Admin can seed via POST /admin/seed with a secret key.)
        # --------------------------------------------------


        # --------------------------------------------------
        # Final Validation
        # --------------------------------------------------
        job.status = "validating"
        db.commit()

        all_scores = [
            s.final_score_after_rewrite or s.combined_score
            for s in span_records
        ]

        all_lengths = [
            len(s.original_text)
            for s in span_records
        ]

        job.global_similarity_score = global_document_score(
            all_scores,
            all_lengths,
        )

        job.status = "done"
        db.commit()

    except Exception as e:
        try:
            db.rollback()
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception as inner_err:
            print(f"Error handling job failure for {job_id}: {inner_err}")

    finally:
        db.close()