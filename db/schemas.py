"""
Pydantic schemas - define the shape of API requests and responses.
"""
from pydantic import BaseModel, model_validator
from typing import Optional, List
import datetime


class JobResponse(BaseModel):
    id: str
    filename: str
    status: str
    global_similarity_score: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class FlaggedSpanResponse(BaseModel):
    """
    Flat span schema aligned to what the Flutter frontend reads:
      - text            <- original_text
      - similarity_score <- combined_score
      - start_offset    <- start_char
      - end_offset      <- end_char
      - source_title    <- matched_source_title
      - rewritten_text  (same name)
    """
    id: str
    text: str                           # Flutter: json['text']
    rewritten_text: Optional[str] = None
    similarity_score: float             # Flutter: json['similarity_score']
    start_offset: int                   # Flutter: json['start_offset']
    end_offset: int                     # Flutter: json['end_offset']
    source_title: Optional[str] = None  # Flutter: json['source_title']
    source_url: Optional[str] = None    # Flutter: json['source_url']
    # Extra detail fields (not used by Flutter yet, but useful for debugging)
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0
    rewrite_attempts: int = 0
    final_score_after_rewrite: Optional[float] = None
    resolved: int = 0

    class Config:
        from_attributes = True

    @classmethod
    def from_db(cls, span) -> "FlaggedSpanResponse":
        """Convert a FlaggedSpan ORM object to this response schema."""
        return cls(
            id=str(span.id),
            text=span.original_text or "",
            rewritten_text=span.rewritten_text,
            similarity_score=span.combined_score or 0.0,
            start_offset=span.start_char or 0,
            end_offset=span.end_char or 0,
            source_title=span.matched_source_title,
            source_url=None,
            lexical_score=span.lexical_score or 0.0,
            semantic_score=span.semantic_score or 0.0,
            combined_score=span.combined_score or 0.0,
            rewrite_attempts=span.rewrite_attempts or 0,
            final_score_after_rewrite=span.final_score_after_rewrite,
            resolved=span.resolved or 0,
        )


class ReportResponse(BaseModel):
    """
    Flat report schema aligned to what the Flutter frontend reads:
      - job_id                  <- job.id
      - filename                <- job.filename
      - global_similarity_score <- job.global_similarity_score
      - flagged_spans           <- spans (list)
      - full_text               (same)
    """
    job_id: str
    filename: str
    status: str
    global_similarity_score: float
    flagged_spans: List[FlaggedSpanResponse]
    full_text: Optional[str] = None
    rewritten_full_text: Optional[str] = None
    created_at: Optional[datetime.datetime] = None
