"""
SQLAlchemy models for users, jobs, document sources, and flagged spans.

User           -> a registered account (email + hashed password)
Job            -> one uploaded document being processed, owned by a User
SourceDocument -> a reference document in the corpus (used to detect plagiarism against)
FlaggedSpan    -> a paragraph/sentence flagged for similarity, with its rewrite history
"""
import uuid
import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config import settings

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    """A registered account. Owns Jobs (uploads/reports)."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")


class Job(Base):
    """Represents one uploaded document and its scan/rewrite lifecycle."""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, default="uploaded")
    # status flow: uploaded -> parsing -> scanning -> rewriting -> validating -> done -> failed

    global_similarity_score = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    full_text = Column(Text, nullable=True)

    user = relationship("User", back_populates="jobs")
    spans = relationship("FlaggedSpan", back_populates="job", cascade="all, delete-orphan")


class SourceDocument(Base):
    """A reference corpus document/sentence used for comparison."""
    __tablename__ = "source_documents"

    id = Column(String, primary_key=True, default=gen_uuid)
    source_title = Column(String, nullable=False)
    sentence_text = Column(Text, nullable=False)
    is_user_upload = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class FlaggedSpan(Base):
    """A specific paragraph in the uploaded doc flagged as similar to a source."""
    __tablename__ = "flagged_spans"

    id = Column(String, primary_key=True, default=gen_uuid)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)

    original_text = Column(Text, nullable=False)
    rewritten_text = Column(Text, nullable=True)

    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)

    matched_source_title = Column(String, nullable=True)
    matched_source_text = Column(Text, nullable=True)
    # Populated only for web-sourced matches (via SearXNG fallback).
    # Stays NULL for matches found in the local SourceDocument corpus.
    source_url = Column(String, nullable=True)

    lexical_score = Column(Float, default=0.0)
    semantic_score = Column(Float, default=0.0)
    combined_score = Column(Float, default=0.0)

    rewrite_attempts = Column(Integer, default=0)
    final_score_after_rewrite = Column(Float, nullable=True)
    resolved = Column(Integer, default=0)  # 0 = still flagged, 1 = below threshold

    job = relationship("Job", back_populates="spans")


# --- Engine & session setup ---
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(engine)

        job_columns = [c['name'] for c in inspector.get_columns('jobs')]
        if 'full_text' not in job_columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN full_text TEXT"))

        span_columns = [c['name'] for c in inspector.get_columns('flagged_spans')]
        if 'source_url' not in span_columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE flagged_spans ADD COLUMN source_url TEXT"))
    except Exception as e:
        print(f"Error checking/adding columns: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()