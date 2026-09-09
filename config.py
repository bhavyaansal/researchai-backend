import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

class Settings:
    # Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/plagiarism.db")

    # GROQ API 
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Embedding model (sentence-transformers — runs locally, free)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 

    # Pipeline thresholds
    SIMILARITY_FLAG_THRESHOLD: float = float(os.getenv("SIMILARITY_FLAG_THRESHOLD", "0.35"))
    TARGET_GLOBAL_THRESHOLD: float = float(os.getenv("TARGET_GLOBAL_THRESHOLD", "0.10"))
    MAX_REWRITE_ATTEMPTS: int = int(os.getenv("MAX_REWRITE_ATTEMPTS", "3"))

    # Hybrid search weighting
    LEXICAL_WEIGHT: float = 0.6
    SEMANTIC_WEIGHT: float = 0.4


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(
    os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "").replace("./", "")) or "data",
    exist_ok=True
)
