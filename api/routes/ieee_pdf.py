"""
ieee_pdf.py

Builds a two-column, IEEE-conference-paper-style PDF that reconstructs
the ORIGINAL document (with AI-rewritten spans merged in) rather than
just dumping flat body text. It detects:
  - a title (the document's own first line, if it looks like one)
  - section headings ("Abstract", "1. Introduction", "I. RELATED WORK", etc.)
and styles them accordingly, so the result reads like an actual paper
the user could submit — not a wall of undifferentiated paragraphs.

Drop this file into api/routes/ (next to download.py) and add
`reportlab` to requirements.txt.
"""
import io
import os
import re
import json
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    NextPageTemplate,
    FrameBreak,
    HRFlowable,
    Image as RLImage,
    Table as RLTable,
    TableStyle,
)
from reportlab.lib import colors

MODIFIED_COLOR = "#0B5FFF"  # blue — used to flag AI-rewritten text

# Matches a block that is ENTIRELY a figure/table placeholder token,
# e.g. "[[FIGURE:fig_0001.png]]" or "[[TABLE:tbl_0001.json]]" — these
# come from ingestion/docx_parser.py's image/table extraction.
_FIGURE_TOKEN_RE = re.compile(r'^\[\[FIGURE:([^\[\]]+)\]\]$')
_TABLE_TOKEN_RE = re.compile(r'^\[\[TABLE:([^\[\]]+)\]\]$')

# --------------------------------------------------------------------
# Heading / title detection heuristics
# --------------------------------------------------------------------

_HEADING_KEYWORDS = {
    "abstract", "keywords", "index terms", "introduction", "related work",
    "literature review", "background", "methodology", "methods",
    "materials and methods", "proposed method", "system design",
    "implementation", "experimental setup", "experiments", "evaluation",
    "results", "results and discussion", "discussion", "conclusion",
    "conclusions", "future work", "limitations", "acknowledgment",
    "acknowledgments", "acknowledgement", "acknowledgements", "references",
    "bibliography", "appendix",
}

# "1. Introduction", "1.2 Related Work", "I. INTRODUCTION", "IV. Results"
_NUMBERED_HEADING_RE = re.compile(
    r'^(\d{1,2}(\.\d{1,2})*\.?|[IVXLCDM]{1,6}\.)\s+[A-Za-z].{0,80}$'
)


def _is_heading(text: str) -> bool:
    """Best-effort detection of a section heading line."""
    t = text.strip()
    if not t or len(t) > 90:
        return False

    bare = t.rstrip(":").strip().lower()
    if bare in _HEADING_KEYWORDS:
        return True

    if _NUMBERED_HEADING_RE.match(t):
        return True

    # Fully caps, short, no terminal sentence punctuation -> likely a heading
    if t.isupper() and 3 <= len(t) <= 60 and not t.endswith((".", ",")):
        return True

    return False


def _looks_like_title(text: str) -> bool:
    """Best-effort detection of a document title (first line of the doc)."""
    t = text.strip()
    if not t or len(t) > 200:
        return False
    if _is_heading(t):
        return False  # e.g. doc starts directly with "Abstract" — no title line
    # Titles don't usually end with sentence-ending punctuation
    if t.endswith((".", ";")):
        return False
    return True


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="IEEETitle", fontName="Times-Bold", fontSize=18, leading=22,
        alignment=TA_CENTER, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="IEEEMeta", fontName="Times-Roman", fontSize=9, leading=12,
        alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="IEEENote", fontName="Times-Roman", fontSize=8.5,
        leading=11, spaceBefore=6, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="IEEEBody", fontName="Times-Roman", fontSize=9.5, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=6, firstLineIndent=12,
    ))
    styles.add(ParagraphStyle(
        name="IEEEHeading", fontName="Times-Bold", fontSize=10.5, leading=13,
        alignment=TA_CENTER, spaceBefore=10, spaceAfter=6,
    ))
    return styles


def reconstruct_document_runs(full_text: str, spans: list) -> list:
    """
    Like reconstruct_document(), but instead of returning a flat string,
    returns an ordered list of (text, is_modified) tuples so the PDF
    builder can visually flag which parts were AI-rewritten.
    """
    if not full_text:
        return []
    sorted_spans = sorted(spans, key=lambda s: s.start_char)
    runs = []
    last_idx = 0
    for span in sorted_spans:
        if span.start_char > last_idx:
            runs.append((full_text[last_idx:span.start_char], False))
        if span.rewritten_text is not None:
            runs.append((span.rewritten_text, True))
        else:
            runs.append((span.original_text, False))
        last_idx = span.end_char
    if last_idx < len(full_text):
        runs.append((full_text[last_idx:], False))
    return runs


def _split_into_blocks(runs):
    """
    Splits the ordered (text, is_modified) runs into paragraph blocks on
    blank-line boundaries. Returns a list of blocks, where each block is
    itself a list of (text, is_modified) runs belonging to that
    paragraph (a run that spans a "\\n\\n" boundary is split across two
    blocks, preserving its is_modified flag on both halves).
    """
    blocks = []
    current = []
    for text, is_modified in runs:
        if not text:
            continue
        parts = text.split("\n\n")
        for i, part in enumerate(parts):
            if part:
                current.append((part, is_modified))
            if i < len(parts) - 1:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return [b for b in blocks if any(t.strip() for t, _ in b)]


def _block_plain_text(block) -> str:
    return "".join(t for t, _ in block).strip()


def _block_to_markup(block) -> str:
    marked = []
    for text, is_modified in block:
        escaped = xml_escape(text)
        if is_modified:
            marked.append(f'<font color="{MODIFIED_COLOR}"><i>{escaped}</i></font>')
        else:
            marked.append(escaped)
    markup = "".join(marked).strip("\n")
    return markup.replace("\n", "<br/>")


def _make_image_flowable(fname: str, assets_dir: str, max_width: float):
    """
    Loads an extracted figure from <assets_dir>/figures/<fname> and
    returns a reportlab Image flowable scaled to fit max_width while
    preserving its original aspect ratio. Returns None if the file is
    missing or unreadable (a missing figure shouldn't crash the whole
    PDF — it just gets skipped).
    """
    if not assets_dir:
        return None
    path = os.path.join(assets_dir, "figures", fname)
    if not os.path.isfile(path):
        return None
    try:
        reader = ImageReader(path)
        orig_w, orig_h = reader.getSize()
        if orig_w <= 0 or orig_h <= 0:
            return None
        scale = min(max_width / orig_w, 1.0)
        # Also cap height so a tall/narrow image can't blow past a
        # column's available vertical space on a single frame.
        display_w = orig_w * scale
        display_h = orig_h * scale
        max_h = 3.2 * inch
        if display_h > max_h:
            shrink = max_h / display_h
            display_w *= shrink
            display_h *= shrink
        return RLImage(path, width=display_w, height=display_h)
    except Exception:
        return None


def _make_table_flowable(fname: str, assets_dir: str, max_width: float, styles):
    """
    Loads an extracted table from <assets_dir>/tables/<fname> (a JSON
    list of row lists) and returns a reportlab Table flowable sized to
    fit max_width. Returns None if the file is missing/unreadable.
    """
    if not assets_dir:
        return None
    path = os.path.join(assets_dir, "tables", fname)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not rows or not rows[0]:
            return None

        num_cols = max(len(r) for r in rows)
        col_width = max_width / num_cols

        # Wrap cell text in Paragraphs so long cell content wraps
        # instead of overflowing the column.
        cell_style = ParagraphStyle(
            name="IEEETableCell", fontName="Times-Roman", fontSize=8, leading=10,
        )
        wrapped_rows = [
            [Paragraph(xml_escape(cell), cell_style) for cell in row]
            for row in rows
        ]

        table = RLTable(wrapped_rows, colWidths=[col_width] * num_cols)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return table
    except Exception:
        return None


def _build_body_flowables(runs, styles, assets_dir=None, col_width=None):
    """
    Turns the raw runs into a list of flowables: Paragraph for text
    (IEEEHeading for detected section headings/titles, IEEEBody for
    everything else), plus real Image/Table flowables wherever a
    [[FIGURE:...]] or [[TABLE:...]] placeholder block is found.

    Returns (title_text_or_None, flowables) — if the very first block
    looks like a document title, it's pulled out separately instead of
    being included in the flowables list.
    """
    blocks = _split_into_blocks(runs)
    if not blocks:
        return None, []

    detected_title = None
    start_index = 0
    first_plain = _block_plain_text(blocks[0])
    if _looks_like_title(first_plain) and not any(m for _, m in blocks[0]):
        # Only trust the original doc's own text as a title (not a
        # rewritten span) — an AI-rewritten first line is unlikely to
        # be the actual paper title.
        detected_title = first_plain
        start_index = 1

    # Fall back to a reasonable width if the caller didn't pass one
    # (e.g. direct/older callers) so image/table sizing still works.
    max_width = col_width or (3.4 * inch)

    flowables = []
    for block in blocks[start_index:]:
        plain = _block_plain_text(block)

        fig_match = _FIGURE_TOKEN_RE.match(plain)
        if fig_match:
            img = _make_image_flowable(fig_match.group(1), assets_dir, max_width)
            if img:
                flowables.append(img)
            continue

        tbl_match = _TABLE_TOKEN_RE.match(plain)
        if tbl_match:
            tbl = _make_table_flowable(tbl_match.group(1), assets_dir, max_width, styles)
            if tbl:
                flowables.append(tbl)
            continue

        markup = _block_to_markup(block)
        if not markup:
            continue
        if _is_heading(plain):
            flowables.append(Paragraph(markup, styles["IEEEHeading"]))
        else:
            flowables.append(Paragraph(markup, styles["IEEEBody"]))

    return detected_title, flowables


def build_ieee_pdf(
    *,
    title: str,
    filename: str,
    job_id: str,
    similarity_pct: float,
    generated_at: str,
    runs: list,
    assets_dir: str = None,
) -> bytes:
    """
    runs: ordered list of (text, is_modified) tuples covering the whole
    document (see reconstruct_document_runs). Returns raw PDF bytes.

    `title` is used as a fallback header title if the document's own
    first line isn't confidently detected as a title (e.g. the doc
    starts directly with "Abstract" or a numbered section).

    `assets_dir` is the per-job folder containing figures/ and tables/
    subfolders (see ingestion/docx_parser.py) — needed to resolve
    [[FIGURE:...]] / [[TABLE:...]] placeholder tokens back into real
    images and tables. If None, any such placeholders are silently
    skipped (e.g. for documents parsed without asset extraction).
    """
    buf = io.BytesIO()
    styles = _build_styles()

    page_w, page_h = LETTER
    margin = 0.6 * inch
    gutter = 0.28 * inch
    col_w = (page_w - 2 * margin - gutter) / 2

    detected_title, body_flowables = _build_body_flowables(
        runs, styles, assets_dir=assets_dir, col_width=col_w
    )
    display_title = detected_title or title

    header_h = 1.5 * inch
    body_top = page_h - margin - header_h

    header_frame = Frame(
        margin, body_top, page_w - 2 * margin, header_h,
        id="header", topPadding=0, bottomPadding=6,
    )
    col_left_first = Frame(
        margin, margin, col_w, body_top - margin, id="colL1", topPadding=6,
    )
    col_right_first = Frame(
        margin + col_w + gutter, margin, col_w, body_top - margin,
        id="colR1", topPadding=6,
    )
    col_left = Frame(
        margin, margin, col_w, page_h - 2 * margin, id="colL", topPadding=6,
    )
    col_right = Frame(
        margin + col_w + gutter, margin, col_w, page_h - 2 * margin,
        id="colR", topPadding=6,
    )

    doc = BaseDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=display_title,
    )
    doc.addPageTemplates([
        PageTemplate(id="First", frames=[header_frame, col_left_first, col_right_first]),
        PageTemplate(id="Later", frames=[col_left, col_right]),
    ])

    story = [
        Paragraph(xml_escape(display_title), styles["IEEETitle"]),
        Paragraph(
            f"Original file: {xml_escape(filename)} &nbsp;&bull;&nbsp; "
            f"Job ID: {xml_escape(job_id)}",
            styles["IEEEMeta"],
        ),
        Paragraph(
            f"Generated by ResearchAI &nbsp;&bull;&nbsp; {xml_escape(generated_at)} "
            f"&nbsp;&bull;&nbsp; Similarity: {similarity_pct:.1f}%",
            styles["IEEEMeta"],
        ),
        HRFlowable(width="100%", thickness=0.75, color=colors.black,
                   spaceBefore=4, spaceAfter=4),
        Paragraph(
            f'<b>Note:</b> Text shown in <font color="{MODIFIED_COLOR}"><i>blue '
            f"italics</i></font> has been AI-rewritten to reduce similarity with "
            f"matched sources. All other text is unchanged from the original "
            f"submission.",
            styles["IEEENote"],
        ),
        FrameBreak(),
        NextPageTemplate("Later"),
    ]

    if not body_flowables:
        body_flowables = [Paragraph("No content available.", styles["IEEEBody"])]
    story.extend(body_flowables)

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes