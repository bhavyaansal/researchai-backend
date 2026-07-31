"""
Pipeline orchestrator — conservative Gemini usage.
Never hangs: every Gemini call should have a timeout inside rewrite_paragraph().
On any failure, keeps original text and continues.
"""

import time

from db.models import (
    SessionLocal,
    Job,
    FlaggedSpan,
    SourceDocument,
)
from ingestion.parser import extract_text, split_into_paragraphs
from search.lexical import build_bm25_index
from search.hybrid import hybrid_search
from mapping.coordinate_mapper import map_span_to_match
from rewriter.rewrite_chain import rewrite_paragraph
from rewriter.validator import validate_rewrite
from scoring.similarity import global_document_score
from search.semantic import index_source_documents
from config import settings


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

        full_text = extract_text(job.file_path)
        job.full_text = full_text

        paragraphs = split_into_paragraphs(full_text)

        # --------------------------------------------------
        # Stage 2 : Scanning
        # --------------------------------------------------
        job.status = "scanning"
        db.commit()

        bm25_index, sources = build_bm25_index(db)

        flagged = []

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
            )

            if (
                match
                and match["combined_score"]
                >= settings.SIMILARITY_FLAG_THRESHOLD
            ):
                flagged.append((para, match))

        # Prevent excessive Gemini requests
        flagged = flagged[:5]

        # --------------------------------------------------
        # Stage 3 : Coordinate Mapping
        # --------------------------------------------------
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
        # Auto-ingest into corpus
        # --------------------------------------------------
        try:
            new_chroma_records = []

            for para in paragraphs:
                text = para["text"].strip()

                if len(text) < 15:
                    continue

                existing = (
                    db.query(SourceDocument)
                    .filter(SourceDocument.sentence_text == text)
                    .first()
                )

                if existing:
                    continue

                src = SourceDocument(
                    source_title=job.filename,
                    sentence_text=text,
                )

                db.add(src)
                db.flush()

                new_chroma_records.append(
                    {
                        "id": str(src.id),
                        "title": job.filename,
                        "text": text,
                    }
                )

            db.commit()

            if new_chroma_records:
                index_source_documents(new_chroma_records)

        except Exception as ingest_err:
            print(f"Warning: Auto-ingest failed: {ingest_err}")

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
        job = db.query(Job).filter(Job.id == job_id).first()

        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()

    finally:
        db.close()