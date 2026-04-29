"""Markdown → PDF exporter for Synthesis reports.

Renders the markdown produced by the Synthesis agent into a styled PDF
following CLAUDE.md §13 colour palette (parchment cover/closing, white body
pages, sage section headers, ink body text, bone dividers between sections).

The exporter is intentionally minimal — it understands only the subset of
markdown the Synthesis prompt produces:
  - One H1 (the report title)
  - H2 section headers (the 7 fixed §11 sections)
  - Paragraphs
  - Unordered bullet lists (`- ` prefix)
  - Ordered numbered lists (`1. `, `2. `, ...)
  - Bold inside a paragraph (`**text**`)

Anything more exotic (tables, images, code fences) is rendered as plain
prose. We don't pull a full markdown library (Mistune / markdown) just for
this — it would be 200KB of dependency for one use site, violating the
minimalism rule in CLAUDE.md §16.2.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

# --- Palette (CLAUDE.md §13) ------------------------------------------------

SAGE = HexColor("#2D4F3A")
PARCHMENT = HexColor("#F4ECDC")
WHITE = HexColor("#FFFFFF")
EGGSHELL = HexColor("#FBF5E8")
TAUPE = HexColor("#E0D5C2")
BONE = HexColor("#EDE5D5")
INK = HexColor("#1A1611")


# --- Styles ------------------------------------------------------------------


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "title": ParagraphStyle(
            "FinaqTitle",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=SAGE,
            spaceAfter=14,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "FinaqSubtitle",
            parent=base,
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceAfter=18,
            alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "FinaqH2",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=20,
            textColor=SAGE,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "FinaqBody",
            parent=base,
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=INK,
            spaceAfter=8,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "FinaqBullet",
            parent=base,
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=INK,
            leftIndent=14,
            bulletIndent=2,
            spaceAfter=4,
        ),
        "numbered": ParagraphStyle(
            "FinaqNumbered",
            parent=base,
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=INK,
            leftIndent=20,
            bulletIndent=2,
            spaceAfter=4,
        ),
    }


# --- Markdown → flowables ----------------------------------------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ESCAPE_RE = re.compile(r"[<>&]")


def _escape_html(s: str) -> str:
    """ReportLab's Paragraph treats input as a mini HTML — `<` and `>` are
    interpreted as tags. Escape them so prose like "P10 < P50" renders
    literally, while preserving the `<b>` we generate for bold."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_inline(text: str) -> str:
    """Apply markdown bold (`**text**`) → ReportLab `<b>text</b>` after
    HTML-escaping the rest of the line."""
    out_parts: list[str] = []
    cursor = 0
    for m in _BOLD_RE.finditer(text):
        out_parts.append(_escape_html(text[cursor : m.start()]))
        out_parts.append(f"<b>{_escape_html(m.group(1))}</b>")
        cursor = m.end()
    out_parts.append(_escape_html(text[cursor:]))
    return "".join(out_parts)


def _split_blocks(markdown: str) -> list[tuple[str, str]]:
    """Tokenise markdown into ('kind', 'content') tuples. Kinds:
    'h1', 'h2', 'subtitle' (the **Date:** line), 'bullet', 'numbered',
    'paragraph', 'blank'."""
    blocks: list[tuple[str, str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            blocks.append(("blank", ""))
            continue
        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
        elif line.startswith("**Date:") or (
            line.startswith("**") and "Confidence" in line and line.endswith("**")
        ):
            blocks.append(("subtitle", line))
        elif line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
        elif re.match(r"^\d+\.\s+", line):
            content = re.sub(r"^\d+\.\s+", "", line).strip()
            blocks.append(("numbered", content))
        else:
            blocks.append(("paragraph", line.strip()))
    return blocks


def _flowables_from_blocks(
    blocks: list[tuple[str, str]], styles: dict[str, ParagraphStyle]
) -> list:
    """Convert tokenised blocks into a list of ReportLab flowables."""
    flow: list = []
    numbered_counter = 0
    in_paragraph_buffer: list[str] = []

    def _flush_paragraph() -> None:
        if in_paragraph_buffer:
            text = " ".join(in_paragraph_buffer)
            flow.append(Paragraph(_render_inline(text), styles["body"]))
            in_paragraph_buffer.clear()

    for kind, content in blocks:
        if kind == "h1":
            _flush_paragraph()
            flow.append(Paragraph(_render_inline(content), styles["title"]))
        elif kind == "subtitle":
            _flush_paragraph()
            flow.append(Paragraph(_render_inline(content), styles["subtitle"]))
        elif kind == "h2":
            _flush_paragraph()
            flow.append(
                HRFlowable(
                    width="100%",
                    thickness=0.6,
                    color=BONE,
                    spaceBefore=8,
                    spaceAfter=2,
                )
            )
            flow.append(Paragraph(_render_inline(content), styles["h2"]))
            numbered_counter = 0
        elif kind == "bullet":
            _flush_paragraph()
            flow.append(
                Paragraph(_render_inline(content), styles["bullet"], bulletText="•")
            )
        elif kind == "numbered":
            _flush_paragraph()
            numbered_counter += 1
            flow.append(
                Paragraph(
                    _render_inline(content),
                    styles["numbered"],
                    bulletText=f"{numbered_counter}.",
                )
            )
        elif kind == "paragraph":
            in_paragraph_buffer.append(content)
        elif kind == "blank":
            _flush_paragraph()

    _flush_paragraph()
    return flow


# --- Page background --------------------------------------------------------


def _on_page(canvas, doc) -> None:
    """Draw a white body page with parchment-tinted top band so the page
    has palette presence without being heavy on ink. Called once per page
    by SimpleDocTemplate."""
    canvas.saveState()
    width, height = LETTER
    # Parchment band along the top edge (0.5in)
    canvas.setFillColor(PARCHMENT)
    canvas.rect(0, height - 0.5 * inch, width, 0.5 * inch, fill=1, stroke=0)
    # Footer accent line
    canvas.setStrokeColor(BONE)
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, 0.55 * inch, width - 0.75 * inch, 0.55 * inch)
    # Footer text
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.75 * inch, 0.4 * inch, "FINAQ — institutional drill-in report")
    canvas.drawRightString(width - 0.75 * inch, 0.4 * inch, f"page {doc.page}")
    canvas.restoreState()


# --- Public API --------------------------------------------------------------


def export(markdown: str, output_path: str | Path) -> Path:
    """Render `markdown` (Synthesis report body) into a PDF at `output_path`.

    Returns the path written. Raises if `markdown` is empty.
    """
    if not markdown or not markdown.strip():
        raise ValueError("Cannot export empty markdown to PDF")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    blocks = _split_blocks(markdown)
    story = _flowables_from_blocks(blocks, styles)

    # Closing parchment band (visual bookend)
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=0.8, color=TAUPE))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.95 * inch,
        bottomMargin=0.85 * inch,
        title="FINAQ drill-in report",
        author="FINAQ",
    )
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return output_path
