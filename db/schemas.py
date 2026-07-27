"""
Pydantic schemas - define the shape of API requests and responses.
"""
from pydantic import BaseModel
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
    id: str
    original_text: str
    rewritten_text: Optional[str] = None
    start_char: int
    end_char: int
    matched_source_title: Optional[str] = None
    lexical_score: float
    semantic_score: float
    combined_score: float
    rewrite_attempts: int
    final_score_after_rewrite: Optional[float] = None
    resolved: int

    class Config:
        from_attributes = True


class ReportResponse(BaseModel):
    job: JobResponse
    spans: List[FlaggedSpanResponse]
    full_text: Optional[str] = None
