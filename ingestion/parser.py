"""
Dispatcher: detects file type by extension and routes to the correct parser.
Also splits the extracted text into paragraphs for downstream processing.
"""
import os
from ingestion.pdf_parser import parse_pdf
from ingestion.docx_parser import parse_docx


class UnsupportedFileTypeError(Exception):
    pass


def extract_text(file_path: str) -> str:
    """Route to the correct parser based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    # elif ext == ".txt":
    #     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    #         return f.read()
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}. Use .pdf, .docx or .txt")


def split_into_paragraphs(text: str) -> list[dict]:
    """
    Split full document text into paragraphs, tracking each paragraph's
    character offset in the original text (needed for coordinate mapping later).

    Returns a list of dicts: [{"text": ..., "start_char": ..., "end_char": ...}, ...]
    """
    if not text:
        return []

    # Normalize CRLF to LF
    text_norm = text.replace("\r\n", "\n")

    paragraphs = []
    cursor = 0

    delimiter = "\n\n" if "\n\n" in text_norm else "\n"
    parts = text_norm.split(delimiter)
    for i, raw_part in enumerate(parts):
        para = raw_part.strip()
        if para:
            start = text_norm.find(para, cursor)
            if start == -1:
                start = cursor
            end = start + len(para)
            paragraphs.append({"text": para, "start_char": start, "end_char": end})
            cursor = end + (len(raw_part) - (start - cursor) - len(para) if start >= cursor else 0)
        cursor += len(delimiter) if i < len(parts) - 1 else 0

    return paragraphs