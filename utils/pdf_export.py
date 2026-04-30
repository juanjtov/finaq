"""Markdown → PDF exporter for Synthesis reports.

Renders the markdown produced by the Synthesis agent into a styled PDF
following CLAUDE.md §13 colour palette (parchment cover/closing, white body
pages, sage section headers, ink body text, bone dividers between sections).

Layout (Step 8 upgrade):
  Page 1 — Cover. Parchment background. TICKER + thesis name (title), date,
           confidence badge, optional KPI table (revenue / FCF yield / P/E /
           current price / shares).
  Page 2+ — Body. White background. The 9 §11 sections rendered from
           markdown, with the Monte Carlo histogram embedded inside the
           "Monte Carlo fair value" section when MC samples are provided.
  Closing — Parchment band as visual bookend.

The exporter is intentionally minimal — it understands only the subset of
markdown the Synthesis prompt produces (H1, H2, paragraphs, `- ` bullets,
`1. ` numbered, `**bold**`). Anything more exotic is rendered as plain
prose. We don't pull a full markdown library — would be 200KB of dependency
for one use site, violating CLAUDE.md §16.2.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils import humanize_amount
from utils.charts import mc_histogram_to_bytes

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
        "cover_title": ParagraphStyle(
            "FinaqCoverTitle",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=32,
            leading=38,
            textColor=SAGE,
            spaceAfter=12,
            alignment=TA_LEFT,
        ),
        "cover_subtitle": ParagraphStyle(
            "FinaqCoverSubtitle",
            parent=base,
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=INK,
            spaceAfter=8,
            alignment=TA_LEFT,
        ),
        "cover_meta": ParagraphStyle(
            "FinaqCoverMeta",
            parent=base,
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=INK,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "title": ParagraphStyle(
            "FinaqTitle",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=26,
            textColor=SAGE,
            spaceAfter=10,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "FinaqSubtitle",
            parent=base,
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceAfter=14,
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
        "footer": ParagraphStyle(
            "FinaqFooter",
            parent=base,
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=INK,
            spaceAfter=2,
            alignment=TA_CENTER,
        ),
    }


# --- Markdown → flowables ----------------------------------------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_URL_RE = re.compile(r"https?://[^\s\)\]]+")


def _escape_html(s: str) -> str:
    """ReportLab's Paragraph treats input as a mini HTML — `<` and `>` are
    interpreted as tags. Escape them so prose like "P10 < P50" renders
    literally, while preserving the `<b>` / `<link>` we inject below."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_inline(text: str) -> str:
    """Apply markdown bold (`**text**`) → ReportLab `<b>text</b>` and convert
    bare URLs into clickable `<link href="...">...</link>` after escaping
    other content. Order matters: bold first (single regex), then URLs on
    the already-escaped output."""
    out_parts: list[str] = []
    cursor = 0
    for m in _BOLD_RE.finditer(text):
        out_parts.append(_linkify(_escape_html(text[cursor : m.start()])))
        out_parts.append(f"<b>{_linkify(_escape_html(m.group(1)))}</b>")
        cursor = m.end()
    out_parts.append(_linkify(_escape_html(text[cursor:])))
    return "".join(out_parts)


def _linkify(escaped: str) -> str:
    """After HTML-escaping, find URLs and wrap them in ReportLab link tags.
    Looking for the escaped form `https:&#47;&#47;...` is wrong because we
    don't escape `/`. URLs survive _escape_html intact (no <,>,& in URLs we
    care about), so we can match plain http(s) here."""
    return _URL_RE.sub(
        lambda m: f'<link href="{m.group(0)}" color="#2D4F3A">{m.group(0)}</link>',
        escaped,
    )


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
    blocks: list[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
    mc_chart_image: Image | None = None,
) -> list:
    """Convert tokenised blocks into ReportLab flowables. When the
    `## Monte Carlo fair value` heading appears AND `mc_chart_image` is
    provided, the chart is inserted right after the heading."""
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
            if (
                mc_chart_image is not None
                and "monte carlo" in content.lower()
            ):
                flow.append(Spacer(1, 0.05 * inch))
                flow.append(mc_chart_image)
                flow.append(Spacer(1, 0.1 * inch))
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


# --- Cover page --------------------------------------------------------------

# Order in which KPIs appear on the cover, with display labels and value
# formatters. Pulled from `state.fundamentals.kpis` if available. Missing
# values are skipped — the table only shows what the run actually has.
# `"humanize_$"` and `"humanize"` are special tokens that delegate to
# utils.humanize_amount for B/M/K compaction; mirrors ui/components._KPI_ROWS.
_COVER_KPIS: tuple[tuple[str, str, str], ...] = (
    ("current_price", "Current price", "${:,.2f}"),
    ("revenue_latest", "Latest revenue", "humanize_$"),
    ("operating_margin_5yr_avg", "Op. margin (5y avg)", "{:.1%}"),
    ("fcf_yield", "FCF yield", "{:.2f}%"),
    ("pe_trailing", "P/E (trailing)", "{:.1f}x"),
    ("revenue_5y_cagr", "Revenue CAGR (5y)", "{:.1%}"),
    ("net_cash", "Net cash", "humanize_$"),
)


def _format_kpi_value(raw: object, fmt: str) -> str:
    """Format a KPI value safely. Returns "—" on errors / missing / NaN."""
    if raw is None:
        return "—"
    if fmt == "humanize_$":
        return humanize_amount(raw, prefix="$")
    if fmt == "humanize":
        return humanize_amount(raw, prefix="")
    try:
        value = float(raw)
        if value != value:  # NaN check
            return "—"
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(raw)


def _kpi_table(kpis: dict, styles: dict[str, ParagraphStyle]) -> Table | None:
    """Render the cover-page KPI table from `state.fundamentals.kpis`. Skips
    KPIs that are missing from the input dict — keeps the visual lean."""
    if not kpis:
        return None
    rows: list[list[Any]] = []
    for key, label, fmt in _COVER_KPIS:
        if key not in kpis or kpis.get(key) is None:
            continue
        value = _format_kpi_value(kpis.get(key), fmt)
        rows.append([
            Paragraph(label, styles["cover_meta"]),
            Paragraph(f"<b>{_escape_html(value)}</b>", styles["cover_meta"]),
        ])
    if not rows:
        return None

    table = Table(rows, colWidths=[2.6 * inch, 2.0 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), EGGSHELL),
            ("BOX", (0, 0), (-1, -1), 0.6, TAUPE),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, BONE),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return table


def _confidence_badge(confidence: str, styles: dict[str, ParagraphStyle]) -> Table:
    """A small one-cell table styled as a "badge" showing low/medium/high.
    Sage fill for high, taupe for medium, bone for low — the hierarchy
    matches the palette's role mapping."""
    confidence = (confidence or "medium").lower()
    fill = {"high": SAGE, "medium": TAUPE, "low": BONE}.get(confidence, TAUPE)
    text_color = "#FFFFFF" if confidence == "high" else "#1A1611"
    table = Table(
        [[Paragraph(
            f'<font color="{text_color}"><b>Confidence: {confidence.upper()}</b></font>',
            styles["cover_meta"],
        )]],
        colWidths=[2.4 * inch],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), fill),
            ("BOX", (0, 0), (-1, -1), 0, fill),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return table


def _build_cover_flowables(
    h1: str,
    subtitle: str,
    confidence: str,
    kpis: dict | None,
    styles: dict[str, ParagraphStyle],
) -> list:
    """Compose the cover page: title, subtitle, confidence badge, KPI table.
    The cover lives on its own page — caller follows with `PageBreak()`."""
    flow: list = []
    flow.append(Spacer(1, 1.4 * inch))
    flow.append(Paragraph(_render_inline(h1), styles["cover_title"]))
    if subtitle:
        flow.append(Paragraph(_render_inline(subtitle), styles["cover_subtitle"]))
    flow.append(Spacer(1, 0.25 * inch))
    flow.append(_confidence_badge(confidence, styles))
    flow.append(Spacer(1, 0.4 * inch))
    table = _kpi_table(kpis or {}, styles)
    if table is not None:
        flow.append(Paragraph("Key metrics", styles["h2"]))
        flow.append(Spacer(1, 0.1 * inch))
        flow.append(table)
    flow.append(Spacer(1, 0.5 * inch))
    flow.append(
        Paragraph(
            "Generated by FINAQ — institutional drill-in research workflow. "
            "Methodology: docs/FINANCE_ASSUMPTIONS.md.",
            styles["footer"],
        )
    )
    return flow


# --- Page background --------------------------------------------------------


def _on_cover_page(canvas, doc) -> None:
    """Cover page is full-bleed parchment with a sage accent stripe."""
    canvas.saveState()
    width, height = LETTER
    canvas.setFillColor(PARCHMENT)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    # Sage accent stripe at top
    canvas.setFillColor(SAGE)
    canvas.rect(0, height - 0.25 * inch, width, 0.25 * inch, fill=1, stroke=0)
    # Footer line on cover
    canvas.setStrokeColor(TAUPE)
    canvas.setLineWidth(0.6)
    canvas.line(0.85 * inch, 0.55 * inch, width - 0.85 * inch, 0.55 * inch)
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.85 * inch, 0.4 * inch, "FINAQ · personal equity research advisor")
    canvas.drawRightString(width - 0.85 * inch, 0.4 * inch, "Cover")
    canvas.restoreState()


def _on_body_page(canvas, doc) -> None:
    """Body pages are white with a parchment-tinted top band + footer."""
    canvas.saveState()
    width, height = LETTER
    canvas.setFillColor(PARCHMENT)
    canvas.rect(0, height - 0.5 * inch, width, 0.5 * inch, fill=1, stroke=0)
    canvas.setStrokeColor(BONE)
    canvas.setLineWidth(0.6)
    canvas.line(0.75 * inch, 0.55 * inch, width - 0.75 * inch, 0.55 * inch)
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.75 * inch, 0.4 * inch, "FINAQ — institutional drill-in report")
    canvas.drawRightString(width - 0.75 * inch, 0.4 * inch, f"page {doc.page - 1}")
    canvas.restoreState()


# --- Helpers ----------------------------------------------------------------


def _extract_h1(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "FINAQ drill-in report"


def _extract_subtitle(markdown: str) -> str:
    """Pull the **Date:** ... **Confidence:** ... line from the markdown."""
    for line in markdown.splitlines():
        if line.startswith("**Date:"):
            # Strip the bold markers for cover display
            text = line.replace("**", "")
            return text
    return ""


def _extract_confidence(markdown: str) -> str:
    """Return low/medium/high from the markdown header, or 'medium' if absent."""
    m = re.search(r"\*\*Confidence:\*\*\s+(low|medium|high)", markdown, re.IGNORECASE)
    return m.group(1).lower() if m else "medium"


def _strip_h1_and_header_subtitle(markdown: str) -> str:
    """Drop the H1 + the Date/Confidence subtitle line so the body starts
    with the first H2 (## What this means). The cover already shows the H1
    and date/confidence; repeating them on page 2 looks unprofessional."""
    out_lines: list[str] = []
    saw_h1 = False
    for line in markdown.splitlines():
        if not saw_h1 and line.startswith("# "):
            saw_h1 = True
            continue
        if saw_h1 and line.startswith("**Date:"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).lstrip()


def _build_mc_image(
    mc_samples: Sequence[float] | None,
    current_price: float | None,
) -> Image | None:
    """Render the MC histogram into an Image flowable. None → no chart embed
    (caller passed nothing)."""
    if mc_samples is None or len(list(mc_samples)) == 0:
        return None
    png = mc_histogram_to_bytes(mc_samples, current_price=current_price)
    img = Image(io.BytesIO(png), width=6.5 * inch, height=3.25 * inch)
    img.hAlign = "CENTER"
    return img


# --- Public API --------------------------------------------------------------


def export(
    markdown: str,
    output_path: str | Path,
    *,
    mc_samples: Sequence[float] | None = None,
    current_price: float | None = None,
    kpis: dict | None = None,
    confidence: str | None = None,
) -> Path:
    """Render `markdown` into a styled PDF at `output_path`.

    Optional kwargs (all backwards-compatible — calling with just markdown
    still works):
      - mc_samples: numpy-array-like of fair-value samples. When provided,
        a palette-aware histogram is embedded inside the "Monte Carlo fair
        value" section.
      - current_price: marks the current-price line on the histogram.
      - kpis: dict of `state.fundamentals.kpis` to render as a styled table
        on the cover page.
      - confidence: low/medium/high override. Defaults to whatever the
        markdown's `**Confidence:**` line says.

    Returns the path written. Raises if `markdown` is empty.
    """
    if not markdown or not markdown.strip():
        raise ValueError("Cannot export empty markdown to PDF")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()

    # Cover page
    h1 = _extract_h1(markdown)
    subtitle = _extract_subtitle(markdown)
    chosen_confidence = (confidence or _extract_confidence(markdown)).lower()
    if chosen_confidence not in ("low", "medium", "high"):
        chosen_confidence = "medium"
    cover_flowables = _build_cover_flowables(h1, subtitle, chosen_confidence, kpis, styles)

    # Body
    body_md = _strip_h1_and_header_subtitle(markdown)
    blocks = _split_blocks(body_md)
    mc_image = _build_mc_image(mc_samples, current_price)
    body_flowables = _flowables_from_blocks(blocks, styles, mc_chart_image=mc_image)

    # Closing parchment band (visual bookend)
    body_flowables.append(Spacer(1, 0.3 * inch))
    body_flowables.append(HRFlowable(width="100%", thickness=0.8, color=TAUPE))

    story = cover_flowables + [PageBreak()] + body_flowables

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
    doc.build(
        story,
        onFirstPage=_on_cover_page,
        onLaterPages=_on_body_page,
    )
    return output_path
