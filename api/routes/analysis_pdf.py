"""
api/routes/analysis_pdf.py
--------------------------
Builds the "Academic Integrity & Plagiarism Analysis" PDF (cover metrics, top
sources table, highlighted document text, segment-by-segment breakdown).

Only dependency: reportlab (already in requirements.txt).

Usage:
    from .analysis_pdf import build_report_pdf, job_to_report
    pdf_bytes = build_report_pdf(job_to_report(job, spans))

See `normalize_report()` for the expected input shape.
"""
from __future__ import annotations

import io
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

# --------------------------------------------------------------------------- #
# Fonts (Vera ships inside reportlab, so nothing extra to install in Docker)
# --------------------------------------------------------------------------- #
_FONT_DIR = os.path.join(os.path.dirname(reportlab.__file__), "fonts")
pdfmetrics.registerFont(TTFont("Body", os.path.join(_FONT_DIR, "Vera.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(_FONT_DIR, "VeraBd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", os.path.join(_FONT_DIR, "VeraIt.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", os.path.join(_FONT_DIR, "VeraBI.ttf")))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic"
)
_GLYPHS = set(pdfmetrics.getFont("Body").face.charToGlyph.keys())

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#dbe3ee")
PANEL = colors.HexColor("#f6f8fb")
DARK = colors.HexColor("#0f172a")

TYPE_STYLE = {
    #            text colour   badge bg      badge border   highlight
    "LEXICAL":  ("#dc2626", "#fee2e2", "#fca5a5", "#fecaca"),
    "HYBRID":   ("#d97706", "#fef3c7", "#fcd34d", "#fde68a"),
    "SEMANTIC": ("#7c3aed", "#f3e8ff", "#d8b4fe", "#e9d5ff"),
}
ORIGINAL_HEX = "#10b981"

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(text) -> str:
    """Escape for reportlab Paragraph and drop glyphs the font can't draw."""
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = "".join(ch if (ord(ch) in _GLYPHS or ch in "\n\t") else "?" for ch in text)
    return escape(text)


def _pct(n, d) -> int:
    return int(round(100.0 * n / d)) if d else 0


def normalize_report(r: dict) -> dict:
    """
    Fill in derived values so callers only have to supply the essentials.

    Expected input (all keys optional except `document_text` / `segments`):
    {
      "document_name": "thesis.docx",
      "generated_at": datetime | str,
      "total_characters": 28648,
      "total_sentences": 256,
      "document_text": "full text ...",
      "segments": [
        {
          "analyzed_text": "sentence from the user's document",
          "source_text":   "matching sentence from the source",
          "source_title":  "Paper title",
          "source_details": "Authors — Venue (url)",
          "match_type":    "SEMANTIC" | "LEXICAL" | "HYBRID",
          "similarity":    0.91          # 0-1 or 0-100, both accepted
        }, ...
      ]
    }
    """
    r = dict(r)
    text = r.get("document_text", "") or ""
    segs = []
    for s in r.get("segments", []):
        s = dict(s)
        sim = float(s.get("similarity", 0) or 0)
        s["similarity"] = int(round(sim * 100 if sim <= 1 else sim))
        s["match_type"] = str(s.get("match_type", "SEMANTIC")).upper()
        if s["match_type"] not in TYPE_STYLE:
            s["match_type"] = "SEMANTIC"
        segs.append(s)
    r["segments"] = segs

    r.setdefault("document_name", "Untitled document")
    ga = r.get("generated_at") or datetime.now()
    if isinstance(ga, datetime):
        ga = ga.strftime("%Y-%m-%d %H:%M:%S")
    r["generated_at"] = ga
    r.setdefault("total_characters", len(text))
    if not r.get("total_sentences"):
        r["total_sentences"] = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))

    counts = {t: 0 for t in TYPE_STYLE}
    for s in segs:
        counts[s["match_type"]] += int(s.get("sentence_count", 1) or 1)
    matched = sum(counts.values())
    r["total_sentences"] = total = max(r["total_sentences"], matched)
    r["counts"] = counts
    r["original_count"] = max(0, total - matched)
    r["plagiarism_score"] = int(round(r["plagiarism_score"])) if r.get("plagiarism_score") is not None \
        else _pct(matched, total)

    # Group segments by source for the "Top reference sources" table
    grouped: dict[str, dict] = {}
    for s in segs:
        key = s.get("source_title", "Unknown source")
        g = grouped.setdefault(key, {
            "title": key, "details": s.get("source_details", ""),
            "matches": 0, "types": set(), "max_sim": 0,
        })
        g["matches"] += 1
        g["types"].add(s["match_type"])
        g["max_sim"] = max(g["max_sim"], s["similarity"])
    r["sources"] = sorted(grouped.values(), key=lambda g: (-g["matches"], -g["max_sim"]))
    return r


def _highlight_document(text: str, segments: list[dict]) -> str:
    """Return reportlab-markup with matched sentences highlighted."""
    spans = []
    for s in segments:
        needle = (s.get("analyzed_text") or "").strip()
        if not needle:
            continue
        a, b = s.get("start"), s.get("end")
        if a is not None and b is not None and text[a:b].strip() == needle:
            spans.append((a, b, s["match_type"]))
            continue
        idx = text.find(needle)
        if idx < 0:  # whitespace-tolerant fallback
            pat = r"\s+".join(re.escape(w) for w in needle.split())
            m = re.search(pat, text)
            if not m:
                continue
            idx, end = m.start(), m.end()
        else:
            end = idx + len(needle)
        spans.append((idx, end, s["match_type"]))
    spans.sort()
    out, pos = [], 0
    for a, b, t in spans:
        if a < pos:  # overlapping
            continue
        out.append(_clean(text[pos:a]))
        col = TYPE_STYLE[t][3]
        out.append(f'<font backColor="{col}">{_clean(text[a:b])}</font>')
        pos = b
    out.append(_clean(text[pos:]))
    return "".join(out).replace("\n", "<br/>")


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
class _NumberedCanvas(rl_canvas.Canvas):
    footer_left = "ResearchAI Plagiarism Detection Platform"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.setFont("Body", 7.5)
            self.setFillColor(MUTED)
            self.setStrokeColor(LINE)
            self.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
            self.drawString(MARGIN, 9.5 * mm, self.footer_left)
            self.drawRightString(PAGE_W - MARGIN, 9.5 * mm,
                                 f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _styles():
    S = {}
    S["title"] = ParagraphStyle("title", fontName="Body-Bold", fontSize=22, leading=27, textColor=INK)
    S["eyebrow"] = ParagraphStyle("eyebrow", fontName="Body-Bold", fontSize=7.5, leading=10,
                                  textColor=colors.HexColor("#5b6b8a"))
    S["h2"] = ParagraphStyle("h2", fontName="Body-Bold", fontSize=12, leading=16, textColor=INK,
                             spaceBefore=6)
    S["meta_k"] = ParagraphStyle("meta_k", fontName="Body-Bold", fontSize=8, leading=11, textColor=INK)
    S["meta_v"] = ParagraphStyle("meta_v", fontName="Body", fontSize=8, leading=11, textColor=INK)
    S["card_l"] = ParagraphStyle("card_l", fontName="Body-Bold", fontSize=6.5, leading=9,
                                 alignment=TA_CENTER, textColor=MUTED)
    S["card_l_dark"] = ParagraphStyle("card_l_dark", parent=S["card_l"], textColor=colors.HexColor("#cbd5e1"))
    S["card_n"] = ParagraphStyle("card_n", fontName="Body-Bold", fontSize=22, leading=26, alignment=TA_CENTER)
    S["card_s"] = ParagraphStyle("card_s", fontName="Body", fontSize=6.5, leading=9, alignment=TA_CENTER,
                                 textColor=MUTED)
    S["th"] = ParagraphStyle("th", fontName="Body-Bold", fontSize=7.5, leading=10, textColor=INK)
    S["td"] = ParagraphStyle("td", fontName="Body", fontSize=8, leading=11, textColor=INK)
    S["td_b"] = ParagraphStyle("td_b", fontName="Body-Bold", fontSize=8.5, leading=11.5, textColor=INK)
    S["td_s"] = ParagraphStyle("td_s", fontName="Body", fontSize=6.8, leading=9, textColor=MUTED)
    S["doc"] = ParagraphStyle("doc", fontName="Body", fontSize=8, leading=12, textColor=colors.HexColor("#1f2937"))
    S["seg_t"] = ParagraphStyle("seg_t", fontName="Body-Bold", fontSize=8.5, leading=11, textColor=INK)
    S["seg_ref"] = ParagraphStyle("seg_ref", fontName="Body", fontSize=7.2, leading=10, textColor=colors.HexColor("#334155"))
    S["seg_lbl"] = ParagraphStyle("seg_lbl", fontName="Body-Bold", fontSize=7, leading=9, textColor=colors.HexColor("#475569"))
    S["seg_txt"] = ParagraphStyle("seg_txt", fontName="Body", fontSize=7.6, leading=11, textColor=colors.HexColor("#1f2937"))
    return S


def _badge(text, t, S):
    fg, bg, border, _ = TYPE_STYLE[t]
    p = Paragraph(f'<font color="{fg}">{_clean(text)}</font>',
                  ParagraphStyle("b", fontName="Body-Bold", fontSize=6.5, leading=8, alignment=TA_CENTER))
    tb = Table([[p]], colWidths=[None])
    tb.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(border)),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tb


def _section(title, S):
    return [
        Paragraph(_clean(title.upper()), S["h2"]),
        Table([[""]], colWidths=[CONTENT_W], rowHeights=[2],
              style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, LINE)])),
        Spacer(1, 6),
    ]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _cover(r, S):
    story = [Paragraph("Academic Integrity &amp; Plagiarism Analysis", S["title"]), Spacer(1, 2),
             Paragraph(_clean(r.get("brand", "RESEARCHAI PLAGIARISM DETECTION ENGINE").upper()), S["eyebrow"]),
             Spacer(1, 10)]
    k, v = S["meta_k"], S["meta_v"]
    meta = Table([
        [Paragraph("Analyzed Document:", k), Paragraph(_clean(r["document_name"]), v),
         Paragraph("Date Generated:", k), Paragraph(_clean(r["generated_at"]), v)],
        [Paragraph("Total Characters:", k), Paragraph(f'{r["total_characters"]:,}', v),
         Paragraph("Total Sentences:", k), Paragraph(f'{r["total_sentences"]:,}', v)],
    ], colWidths=[CONTENT_W * .21, CONTENT_W * .29, CONTENT_W * .20, CONTENT_W * .30])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("ROUNDEDCORNERS", [5, 5, 5, 5]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [meta, Spacer(1, 16)]

    story += _section("Integrity Summary", S)
    tot = r["total_sentences"]
    c = r["counts"]
    cards = [
        ("PLAGIARISM<br/>SCORE", f'{r["plagiarism_score"]}%', "#ffffff", None, True),
        ("LEXICAL<br/>MATCHES", f'{_pct(c["LEXICAL"], tot)}%', TYPE_STYLE["LEXICAL"][0], c["LEXICAL"], False),
        ("HYBRID<br/>MATCHES", f'{_pct(c["HYBRID"], tot)}%', TYPE_STYLE["HYBRID"][0], c["HYBRID"], False),
        ("SEMANTIC<br/>MATCHES", f'{_pct(c["SEMANTIC"], tot)}%', TYPE_STYLE["SEMANTIC"][0], c["SEMANTIC"], False),
        ("ORIGINAL<br/>CONTENT", f'{_pct(r["original_count"], tot)}%', ORIGINAL_HEX, r["original_count"], False),
    ]
    gap = 4 * mm
    w = (CONTENT_W - gap * 4) / 5
    cells, widths = [], []
    for i, (lbl, num, col, n, dark) in enumerate(cards):
        inner = [[Paragraph(lbl, S["card_l_dark" if dark else "card_l"])],
                 [Paragraph(f'<font color="{col}">{num}</font>', S["card_n"])]]
        if n is not None:
            inner.append([Paragraph(f"{n} sentence(s)", S["card_s"])])
        t = Table(inner, colWidths=[w])
        st = [("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
              ("BOX", (0, 0), (-1, -1), 0.6, DARK if dark else LINE), ("ROUNDEDCORNERS", [6, 6, 6, 6])]
        if dark:
            st.append(("BACKGROUND", (0, 0), (-1, -1), DARK))
        t.setStyle(TableStyle(st))
        cells.append(t)
        widths.append(w)
        if i < 4:
            cells.append("")
            widths.append(gap)
    row = Table([cells], colWidths=widths)
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [row, Spacer(1, 18)]
    return story


def _sources_table(r, S):
    story = _section("Top Reference Sources Matched", S)
    if not r["sources"]:
        story.append(Paragraph("No matching sources were found.", S["td"]))
        return story
    rows = [[Paragraph(h, S["th"]) for h in ("#", "Source Document Details", "Matches", "Match Type", "Max Sim")]]
    for i, g in enumerate(r["sources"], 1):
        types = sorted(g["types"])
        badges = [_badge(t, t, S) for t in types]
        badge_cell = badges[0] if len(badges) == 1 else Table([[b] for b in badges])
        rows.append([
            Paragraph(str(i), S["td"]),
            [Paragraph(_clean(g["title"]), S["td_b"]), Spacer(1, 2), Paragraph(_clean(g["details"]), S["td_s"])],
            Paragraph(str(g["matches"]), S["td"]),
            badge_cell,
            Paragraph(f'<b>{g["max_sim"]}%</b>', S["td"]),
        ])
    cw = [CONTENT_W * f for f in (.05, .55, .10, .18, .12)]
    t = Table(rows, colWidths=cw, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), PANEL), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafbfd")))
    t.setStyle(TableStyle(style))
    story.append(t)
    return story


def _document_content(r, S):
    story = _section("Document Content Analysis", S)
    html = _highlight_document(r.get("document_text", ""), r["segments"])
    # Split on line breaks into rows so the bordered box can flow across pages
    paras = [p for p in html.split("<br/>")] or [""]
    rows = [[Paragraph(p if p.strip() else "&nbsp;", S["doc"])] for p in paras]
    t = Table(rows, colWidths=[CONTENT_W], splitByRow=1)
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
    ]))
    story.append(t)
    return story


def _segment_card(i, s, S):
    header = Table([[Paragraph(f"Segment #{i}", S["seg_t"]),
                     _badge(f'{s["match_type"]} MATCH', s["match_type"], S),
                     _badge(f'{s["similarity"]}% SIMILARITY', s["match_type"], S)]],
                   colWidths=[CONTENT_W - 16 - 70 * mm, 34 * mm, 36 * mm])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    ref_txt = f'<b>Reference Source:</b> {_clean(s.get("source_title", ""))}'
    if s.get("source_details"):
        ref_txt += f' ({_clean(s["source_details"])})'
    inner_w = CONTENT_W - 16
    half = inner_w / 2 - 4
    cols = Table([
        [Paragraph("ANALYZED TEXT", S["seg_lbl"]), Paragraph("SOURCE TEXT", S["seg_lbl"])],
        [Paragraph(_clean(s.get("analyzed_text", "")), S["seg_txt"]),
         Paragraph(_clean(s.get("source_text", "")), S["seg_txt"])],
    ], colWidths=[half, half])
    cols.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (-1, 1), PANEL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, 1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
        ("LINEAFTER", (0, 1), (0, 1), 6, colors.white),
    ]))
    card = Table([[header], [Paragraph(ref_txt, S["seg_ref"])], [cols]], colWidths=[CONTENT_W])
    card.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("ROUNDEDCORNERS", [6, 6, 6, 6]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return KeepTogether([card, Spacer(1, 8)])


def _segments(r, S):
    story = _section("Segment-by-Segment Matching Breakdown", S)
    if not r["segments"]:
        story.append(Paragraph("No matched segments.", S["td"]))
    for i, s in enumerate(r["segments"], 1):
        story.append(_segment_card(i, s, S))
    return story


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_report_pdf(report: dict, footer_left: str | None = None) -> bytes:
    r = normalize_report(report)
    S = _styles()
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=20 * mm,
        title=f'Plagiarism Report - {r["document_name"]}', author="ResearchAI",
    )
    doc.addPageTemplates([PageTemplate(
        id="p", frames=[Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - MARGIN - 20 * mm,
                              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)])])
    story = _cover(r, S) + _sources_table(r, S)
    from reportlab.platypus import PageBreak
    story += [PageBreak()] + _document_content(r, S) + [PageBreak()] + _segments(r, S)

    canvas_cls = _NumberedCanvas
    if footer_left:
        canvas_cls = type("C", (_NumberedCanvas,), {"footer_left": footer_left})
    doc.build(story, canvasmaker=canvas_cls)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Adapter: ResearchAI Job + FlaggedSpan rows  ->  report dict
# --------------------------------------------------------------------------- #
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")


def _count_sentences(t: str) -> int:
    return max(1, len(_SENT_RE.findall(t or "")))


def classify_span(lexical: float, semantic: float) -> str:
    """
    Map your two detector scores (0-1) to a match type.
    Tune the 0.5 cut-off to match how your pipeline treats a "hit".
    """
    lex, sem = (lexical or 0.0), (semantic or 0.0)
    if lex >= 0.5 and sem >= 0.5:
        return "HYBRID"
    return "LEXICAL" if lex >= sem else "SEMANTIC"


def job_to_report(job, spans) -> dict:
    """`job` = db.models.Job, `spans` = list[db.models.FlaggedSpan] (duck-typed)."""
    full_text = job.full_text or ""
    segments = []
    for sp in sorted(spans, key=lambda x: x.start_char):
        segments.append({
            "analyzed_text": sp.original_text,
            "source_text": sp.matched_source_text or "",
            "source_title": sp.matched_source_title or "Unknown Source",
            "source_details": sp.source_url or "Local reference corpus",
            "match_type": classify_span(sp.lexical_score, sp.semantic_score),
            "similarity": sp.combined_score or 0.0,
            "sentence_count": _count_sentences(sp.original_text),
            "start": sp.start_char,
            "end": sp.end_char,
        })
    return {
        "document_name": job.filename or "Untitled document",
        "generated_at": job.updated_at or job.created_at,
        "document_text": full_text,
        "total_characters": len(full_text),
        "total_sentences": _count_sentences(full_text),
        "plagiarism_score": (job.global_similarity_score or 0.0) * 100,
        "segments": segments,
    }
