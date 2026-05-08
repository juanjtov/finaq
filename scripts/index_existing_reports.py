"""Backfill the `synthesis_reports` ChromaDB collection from past drill-in
reports on disk (`data_cache/demos/*.json`).

The CIO planner (Step 11.8) does RAG over past drill-ins to support its
"reuse vs drill" decision: if a recent report already covers what the
heartbeat would surface, the CIO chooses `action=reuse` instead of paying
for another full graph run. That requires every past drill-in's Synthesis
markdown to be retrievable by ticker / thesis / theme.

This script:
  1. Globs `data_cache/demos/*.json` (and any future report dump locations).
  2. Splits the `report` markdown on H2 headers — each section becomes a
     chunk so the planner can retrieve at the section level (e.g. the
     "Top risks" section without the bull bullets).
  3. Embeds + upserts into the `synthesis_reports` collection. Idempotent:
     re-running upserts the same chunk ids and overwrites in place.

Filename convention:
  `{TICKER}__{thesis_slug}__{hash}.json`  (newer drill-ins)
  `{TICKER}__{thesis_slug}.json`         (older / single-run demos)

Each chunk's `id` is `f"{run_id}-{section_slug}"` where `run_id` is the
filename stem (always unique within `data_cache/demos/`). Metadata holds
ticker, thesis, section, date, confidence, risk_level — the fields the
planner needs for its `where`-clause prefilter.

Usage:
  python -m scripts.index_existing_reports        # backfill everything
  python -m scripts.index_existing_reports --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Bootstrap so this can be run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.chroma import _get_collection
from utils import logger

REPORTS_DIR = Path("data_cache/demos")
COLLECTION_NAME = "synthesis_reports"

# Section split: H2 headers of the synthesis Markdown (`## What this means`, ...).
# `(?m)` so `^` matches any line start. `(?P<title>...)` captures the heading.
_H2_RE = re.compile(r"(?m)^##\s+(?P<title>[^\n]+)\s*$")

# Date stamp inside the report body: `**Date:** 2026-05-05`.
_DATE_RE = re.compile(r"\*\*Date:\*\*\s+(\d{4}-\d{2}-\d{2})")


def _slugify_section(title: str) -> str:
    """Filename-safe section slug. 'What this means' → 'what_this_means'."""
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s or "section"


def _split_report_into_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a Synthesis report into (section_title, body) tuples.

    Body is the markdown between this H2 and the next H2 (or EOF). The
    H1 + preamble (Date / Confidence) attaches to the first section as
    a head note so the chunk is self-contained.
    """
    matches = list(_H2_RE.finditer(markdown))
    if not matches:
        # No H2 headers — treat the whole thing as one chunk.
        return [("Full report", markdown.strip())]

    sections: list[tuple[str, str]] = []
    head = markdown[: matches[0].start()].strip()  # H1 + Date + Confidence
    for i, m in enumerate(matches):
        title = m.group("title").strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        # Prepend the head note to the first section so retrieval surfaces
        # the date + confidence alongside the content.
        if i == 0 and head:
            body = f"{head}\n\n{body}"
        sections.append((title, body))
    return sections


def _parse_report_filename(stem: str) -> tuple[str, str | None]:
    """Extract (ticker, thesis_slug) from `{TICKER}__{thesis}[__{hash}].json` stems.

    Falls back to (stem, None) on unrecognised pattern.
    """
    parts = stem.split("__")
    if len(parts) < 2:
        return stem, None
    ticker = parts[0].upper()
    thesis = parts[1]
    return ticker, thesis


def _safe_get(d: dict, *path, default=None):
    cur: object = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_metadata(
    *,
    run_id: str,
    ticker: str,
    thesis: str | None,
    section: str,
    data: dict,
    file_mtime: str,
) -> dict:
    """ChromaDB metadata for a single section chunk.

    All values must be primitives (str / int / float / bool / None) — Chroma
    rejects nested dicts. None values are dropped (Chroma also rejects None).
    """
    rep = data.get("report") or ""
    date_match = _DATE_RE.search(rep)
    date = date_match.group(1) if date_match else file_mtime[:10]

    confidence = data.get("synthesis_confidence")
    risk_level = _safe_get(data, "risk", "level")
    mc_p50 = _safe_get(data, "monte_carlo", "p50") or _safe_get(
        data, "monte_carlo", "dcf", "p50"
    )

    meta: dict[str, str | float] = {
        "run_id": run_id,
        "ticker": ticker.upper(),
        "section": section,
        "date": date,
        "filed_at_iso": file_mtime,
    }
    if thesis:
        meta["thesis"] = thesis
    if confidence:
        meta["confidence"] = str(confidence)
    if risk_level:
        meta["risk_level"] = str(risk_level)
    if mc_p50 is not None:
        try:
            meta["mc_p50"] = float(mc_p50)
        except (TypeError, ValueError):
            pass
    return meta


def index_one(path: Path, *, dry_run: bool = False) -> int:
    """Index a single report JSON. Returns the chunk count written (or planned)."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"[index_existing_reports] {path.name}: parse failed: {e}")
        return 0

    report = data.get("report") or ""
    if not report.strip():
        logger.info(f"[index_existing_reports] {path.name}: empty report, skipping")
        return 0

    run_id = path.stem
    ticker, thesis = _parse_report_filename(path.stem)
    file_mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

    sections = _split_report_into_sections(report)
    if not sections:
        return 0

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for title, body in sections:
        if not body.strip():
            continue
        section_slug = _slugify_section(title)
        ids.append(f"{run_id}-{section_slug}")
        docs.append(body)
        metas.append(
            _build_metadata(
                run_id=run_id,
                ticker=ticker,
                thesis=thesis,
                section=title,
                data=data,
                file_mtime=file_mtime,
            )
        )

    if not docs:
        return 0

    if dry_run:
        logger.info(
            f"[index_existing_reports] DRY-RUN {path.name}: would upsert "
            f"{len(docs)} chunks ({ticker} / {thesis})"
        )
        return len(docs)

    coll = _get_collection(name=COLLECTION_NAME)
    coll.upsert(ids=ids, documents=docs, metadatas=metas)
    logger.info(
        f"[index_existing_reports] {path.name}: indexed {len(docs)} chunks "
        f"({ticker} / {thesis})"
    )
    return len(docs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse + chunk but don't write to ChromaDB.",
    )
    parser.add_argument(
        "--reports-dir", default=str(REPORTS_DIR),
        help="Override the source directory (default: data_cache/demos).",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        logger.error(f"[index_existing_reports] {reports_dir} does not exist")
        return 1

    paths = sorted(reports_dir.glob("*.json"))
    if not paths:
        logger.warning(f"[index_existing_reports] no JSON reports in {reports_dir}")
        return 0

    total = 0
    for p in paths:
        total += index_one(p, dry_run=args.dry_run)

    verb = "would upsert" if args.dry_run else "upserted"
    logger.info(
        f"[index_existing_reports] done: {verb} {total} chunks across "
        f"{len(paths)} report(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
