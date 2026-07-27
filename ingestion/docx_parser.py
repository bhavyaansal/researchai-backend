"""
DOCX parsing using python-docx.
Extracts paragraph text, skipping empty lines.
"""
from docx import Document


def parse_docx(file_path: str) -> str:
    """
    Extract all paragraph text from a Word document.
    Joins paragraphs with double newlines to preserve structure.
    """
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
