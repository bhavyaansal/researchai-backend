"""
Local filesystem storage - free replacement for S3/MinIO.
Saves uploaded files to disk under UPLOAD_DIR.
"""
import os
import uuid
from config import settings


def save_upload(file_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """
    Save uploaded file bytes to disk with a unique name to avoid collisions.
    Returns (job_id, saved_file_path).
    """
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(original_filename)[1]
    safe_name = f"{job_id}{ext}"
    full_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    with open(full_path, "wb") as f:
        f.write(file_bytes)

    return job_id, full_path
