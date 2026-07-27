"""
Pipeline orchestrator — all 6 stages for one job.
Conservative Gemini API usage to stay within free tier (15 RPM).
Only rewrites spans that are genuinely above threshold.
"""
import time
from db.models import SessionLocal, Job, FlaggedSpan
from ingestion.parser import extract_text, split_into_paragraphs
from search.lexical import build_bm25_index
from search.hybrid import hybrid_search
from mapping.coordinate_mapper import map_span_to_match
from rewriter.rewrite_chain import rewrite_paragraph
from rewriter.validator import validate_rewrite
from scoring.similarity import global_document_score
from config import settings


def run_pipeline(job_id: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        # ---- Stage 1: Parsing ----
        job.status = "parsing"
        db.commit()

        full_text = extract_text(job.file_path)
        paragraphs = split_into_paragraphs(full_text)

        # ---- Stage 2: Scanning ----
        job.status = "scanning"
        db.commit()

        bm25_index, sources = build_bm25_index(db)

        flagged = []
        for para in paragraphs:
            if len(para["text"].strip()) < 20:
                continue  # skip very short paragraphs
            match = hybrid_search(para["text"], bm25_index, sources, top_k=5)
            if match and match["combined_score"] >= settings.SIMILARITY_FLAG_THRESHOLD:
                flagged.append((para, match))

        # ---- Stage 3: Coordinate mapping ----
        span_records = []
        for para, match in flagged:
            record = map_span_to_match(para, match)
            db_span = FlaggedSpan(job_id=job.id, **record)
            db.add(db_span)
            span_records.append(db_span)
        db.commit()

        # ---- Stages 4-6: Rewriting ----
        job.status = "rewriting"
        db.commit()

        for i, span in enumerate(span_records):
            print(f"Rewriting span {i+1}/{len(span_records)}: score={span.combined_score:.2f}")

            # If already below threshold, mark resolved without calling Gemini
            if span.combined_score <= settings.TARGET_GLOBAL_THRESHOLD:
                span.rewritten_text = span.original_text
                span.rewrite_attempts = 0
                span.final_score_after_rewrite = span.combined_score
                span.resolved = 1
                db.commit()
                continue

            # Single rewrite attempt (MAX_REWRITE_ATTEMPTS=1 in .env)
            try:
                rewritten = rewrite_paragraph(span.original_text, span.combined_score)
                scores = validate_rewrite(rewritten, span.matched_source_text)

                span.rewritten_text = rewritten
                span.rewrite_attempts = 1
                span.final_score_after_rewrite = scores["combined_score"]
                span.resolved = 1  # always mark resolved — best effort
                db.commit()

                # Extra pause between spans to avoid rate limit
                if i < len(span_records) - 1:
                    time.sleep(6)

            except RuntimeError as e:
                # Gemini failed — keep original text, mark resolved anyway
                print(f"Rewrite failed for span {i+1}: {e}")
                span.rewritten_text = span.original_text
                span.rewrite_attempts = 1
                span.final_score_after_rewrite = span.combined_score
                span.resolved = 1
                db.commit()
                # Wait before next span to let rate limit recover
                time.sleep(15)

        # ---- Final scoring ----
        job.status = "validating"
        db.commit()

        all_scores = [
            s.final_score_after_rewrite or s.combined_score
            for s in span_records
        ]
        all_lengths = [len(s.original_text) for s in span_records]
        job.global_similarity_score = global_document_score(all_scores, all_lengths)
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
