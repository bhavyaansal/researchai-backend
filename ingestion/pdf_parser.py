"""
PDF parsing using PyMuPDF (fitz).
Extracts plain text, preserving paragraph breaks reasonably well.
"""
import fitz  # PyMuPDF


def parse_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file, joining pages with double newlines
    so paragraph structure is roughly preserved for later sentence splitting.
    """
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)
