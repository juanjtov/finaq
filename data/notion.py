"""Notion as FINAQ's long-term memory of record (Step 9).

Five functions, all env-var-gated — if `NOTION_API_KEY` isn't set every call
is a no-op so the rest of FINAQ works exactly as today (cached state on
disk, dashboard renders, etc.). Notion is **content** memory, distinct
from `data_cache/state.db` (operations) and ChromaDB (vector store).

Public API
----------
read_thesis_notes(slug)
    Pull the user's free-text annotations for a thesis. Phase 1 Triage
    folds these into material-threshold scoring.
write_report(ticker, thesis, markdown, ...)
    Persist a synthesis report as a row in the Reports DB with the markdown
    rendered as Notion blocks (heading / paragraph / bullet / numbered).
read_watchlist(thesis)
    Pull any standing watchlist items the user has marked active. Triage
    uses these as additional monitoring rules.
write_alert(...)
    Phase 1 Triage entry point — creates an alert row with status='pending'.
update_alert_status(alert_id, status)
    Phase 1 Triage / dashboard / Telegram bot — flip alert status.

Configuration
-------------
The 4 database IDs are read from `.env`:
  NOTION_DB_THESES, NOTION_DB_REPORTS, NOTION_DB_ALERTS, NOTION_DB_WATCHLIST
plus the API token: NOTION_API_KEY

Run `python -m scripts.bootstrap_notion` once with the key set to create the
DBs in your workspace and print the IDs to paste back.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

from utils import logger

# --- Module-level state -----------------------------------------------------

# Single shared client per process. None when NOTION_API_KEY isn't set —
# every public function returns a no-op result in that case.
_client_cache: Any | None = None


def _get_client() -> Any | None:
    """Return a cached `notion_client.Client`, or None if unconfigured.

    We cache because constructing the client is cheap but the requests
    Session inside it benefits from connection reuse across many writes
    in a single drill-in (one report write + multiple block uploads).
    """
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    api_key = os.environ.get("NOTION_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from notion_client import Client

        _client_cache = Client(auth=api_key)
        return _client_cache
    except Exception as e:  # pragma: no cover — import-time / network safety
        logger.warning(f"[notion] could not construct client: {e}")
        return None


def is_configured() -> bool:
    """True if `NOTION_API_KEY` is set. Cheap predicate for the runner +
    Mission Control to avoid issuing no-op writes when Notion isn't wired."""
    return bool(os.environ.get("NOTION_API_KEY", "").strip())


def _db_id(env_var: str) -> str | None:
    """Resolve a database ID from `.env`. Returns None if missing — caller
    handles the no-op."""
    val = os.environ.get(env_var, "").strip()
    return val or None


# --- Data-source resolution (Notion API 2025-09-03) -----------------------
#
# notion-client 3.x defaults to API version 2025-09-03 which moved query
# semantics from `databases.query(database_id=...)` to
# `data_sources.query(data_source_id=...)`. A database now wraps one or more
# data sources; the "default" data source carries the schema and rows.
# `pages.create({"parent": {"database_id": ...}})` is still backward-compatible
# (it auto-routes to the default data source), so writes don't need updating.
# We cache the lookup per process — it's a single retrieve call per DB.

_data_source_cache: dict[str, str] = {}


def _resolve_data_source_id(db_id: str) -> str | None:
    """Return the default data_source_id for a database, or None if it can't
    be resolved (network error, missing DB, no data sources). Cached per
    process; subsequent calls hit memory."""
    if db_id in _data_source_cache:
        return _data_source_cache[db_id]
    client = _get_client()
    if client is None:
        return None
    try:
        db = client.databases.retrieve(database_id=db_id)
    except Exception as e:
        logger.warning(f"[notion] could not retrieve database {db_id}: {e}")
        return None
    sources = db.get("data_sources") or []
    if not sources:
        logger.warning(f"[notion] database {db_id} has no data_sources")
        return None
    ds_id = sources[0].get("id")
    if ds_id:
        _data_source_cache[db_id] = ds_id
    return ds_id


# --- Markdown → Notion blocks ----------------------------------------------

# Notion's block API has rate limits (~3 requests/second on free tier) and
# a per-page block-creation cap (100 children per call). We chunk uploads.
_BLOCKS_PER_REQUEST = 90  # safely under 100, gives margin for retry

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _rich_text(text: str) -> list[dict]:
    """Convert a string with optional `**bold**` markers into Notion's
    rich_text array. Notion expects a list of segments, each with a
    `type` and `text.content`. Unstyled segments use no annotations;
    bold segments add `annotations.bold = True`."""
    segments: list[dict] = []
    cursor = 0
    for m in _BOLD_RE.finditer(text):
        if m.start() > cursor:
            segments.append({
                "type": "text",
                "text": {"content": text[cursor : m.start()]},
            })
        segments.append({
            "type": "text",
            "text": {"content": m.group(1)},
            "annotations": {"bold": True},
        })
        cursor = m.end()
    if cursor < len(text):
        segments.append({"type": "text", "text": {"content": text[cursor:]}})
    if not segments:
        segments.append({"type": "text", "text": {"content": text}})
    return segments


def _markdown_to_blocks(markdown: str) -> list[dict]:
    """Tokenise the synthesis report markdown into Notion blocks.

    Subset matching `utils/pdf_export.py`'s parser:
      - `# H1`        → heading_1
      - `## H2`       → heading_2
      - `- bullet`    → bulleted_list_item
      - `1. numbered` → numbered_list_item
      - `**Date:** … **Confidence:** …` → quote (the report's subtitle)
      - everything else → paragraph (with `**bold**` inline)
      - blank lines flush a paragraph buffer
    """
    blocks: list[dict] = []
    paragraph_buf: list[str] = []

    def _flush_paragraph() -> None:
        if not paragraph_buf:
            return
        text = " ".join(paragraph_buf).strip()
        if text:
            blocks.append({
                "type": "paragraph",
                "paragraph": {"rich_text": _rich_text(text)},
            })
        paragraph_buf.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            _flush_paragraph()
            continue
        if line.startswith("# "):
            _flush_paragraph()
            blocks.append({
                "type": "heading_1",
                "heading_1": {"rich_text": _rich_text(line[2:].strip())},
            })
        elif line.startswith("## "):
            _flush_paragraph()
            blocks.append({
                "type": "heading_2",
                "heading_2": {"rich_text": _rich_text(line[3:].strip())},
            })
        elif line.startswith("**Date:") or (
            line.startswith("**") and "Confidence" in line and line.endswith("**")
        ):
            _flush_paragraph()
            # Subtitle as a callout-style quote — palette-friendly equivalent
            blocks.append({
                "type": "quote",
                "quote": {"rich_text": _rich_text(line)},
            })
        elif line.startswith("- "):
            _flush_paragraph()
            blocks.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _rich_text(line[2:].strip())},
            })
        elif re.match(r"^\d+\.\s+", line):
            _flush_paragraph()
            content = re.sub(r"^\d+\.\s+", "", line).strip()
            blocks.append({
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": _rich_text(content)},
            })
        else:
            paragraph_buf.append(line.strip())

    _flush_paragraph()
    return blocks


# --- write_report ----------------------------------------------------------


def write_report(
    *,
    ticker: str,
    thesis_name: str,
    markdown: str,
    confidence: str | None = None,
    p50: float | None = None,
    current_price: float | None = None,
    run_id: str | None = None,
) -> str | None:
    """Persist a synthesis report as a new row in the Reports DB.

    Returns the Notion page URL on success, or None when Notion isn't
    configured (no-op). Logs and returns None on any API failure — caller
    must treat this as best-effort and never block on it.
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_REPORTS")
    if client is None or not db_id:
        return None

    today = datetime.now(UTC).date().isoformat()
    title = f"{ticker} — {thesis_name} — {today}"
    properties: dict[str, Any] = {
        "Name": {"title": [{"type": "text", "text": {"content": title}}]},
        "Ticker": {
            "rich_text": [{"type": "text", "text": {"content": ticker.upper()}}]
        },
        "Thesis": {
            "rich_text": [{"type": "text", "text": {"content": thesis_name}}]
        },
        "Date": {"date": {"start": today}},
    }
    if confidence:
        properties["Confidence"] = {"select": {"name": confidence.lower()}}
    if p50 is not None:
        properties["DCF P50"] = {"number": float(p50)}
    if current_price is not None:
        properties["Current Price"] = {"number": float(current_price)}
    if run_id:
        properties["Run ID"] = {
            "rich_text": [{"type": "text", "text": {"content": str(run_id)}}]
        }

    try:
        blocks = _markdown_to_blocks(markdown)
        # Notion caps page-creation children at 100; create the page with
        # the FIRST batch, then append the rest via blocks.children.append.
        first_batch = blocks[:_BLOCKS_PER_REQUEST]
        page = client.pages.create(
            parent={"database_id": db_id},
            properties=properties,
            children=first_batch,
        )
        page_id = page.get("id")
        for offset in range(_BLOCKS_PER_REQUEST, len(blocks), _BLOCKS_PER_REQUEST):
            chunk = blocks[offset : offset + _BLOCKS_PER_REQUEST]
            try:
                client.blocks.children.append(block_id=page_id, children=chunk)
            except Exception as e:
                logger.warning(
                    f"[notion] block append failed at offset {offset} for "
                    f"{title}: {e}"
                )
                break
        return page.get("url")
    except Exception as e:
        logger.error(f"[notion] write_report failed for {title}: {e}")
        return None


# --- read_thesis_notes -----------------------------------------------------


def read_thesis_notes(thesis_slug: str) -> str:
    """Return the page body (as plain text) of the Theses-DB row whose
    `Slug` property matches `thesis_slug`. Phase 1 Triage uses this to
    enrich material-threshold scoring with the user's free-text notes.

    Empty string when Notion isn't configured OR the slug doesn't match
    any row — caller treats both the same way (no notes to fold in).
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_THESES")
    if client is None or not db_id:
        return ""
    ds_id = _resolve_data_source_id(db_id)
    if not ds_id:
        return ""
    try:
        results = client.data_sources.query(
            data_source_id=ds_id,
            filter={
                "property": "Slug",
                "rich_text": {"equals": thesis_slug},
            },
            page_size=1,
        )
        rows = results.get("results", [])
        if not rows:
            return ""
        page_id = rows[0]["id"]
        # Pull the page's children (paragraph blocks etc.) and concatenate
        # plain text. We don't need fidelity here — Triage just wants the
        # user's verbatim notes as a corpus.
        children = client.blocks.children.list(block_id=page_id)
        chunks: list[str] = []
        for block in children.get("results", []):
            block_type = block.get("type")
            text_node = block.get(block_type, {}) if block_type else {}
            for rt in text_node.get("rich_text", []):
                content = rt.get("plain_text") or rt.get("text", {}).get("content") or ""
                if content:
                    chunks.append(content)
            if chunks and not chunks[-1].endswith("\n"):
                chunks.append("\n")
        return "".join(chunks).strip()
    except Exception as e:
        logger.warning(f"[notion] read_thesis_notes failed for {thesis_slug}: {e}")
        return ""


# --- append_thesis_note ----------------------------------------------------


def write_adhoc_thesis(
    *,
    slug: str,
    thesis: Any,  # `utils.schemas.Thesis` — typed as Any to avoid circ import
    db_id: str | None = None,
) -> str | None:
    """Mirror an ad-hoc thesis into a Notion DB row. Best-effort — returns
    the page URL on success, None on no-op / failure.

    Step 10e — Discovery-lite. The existing Theses DB is for curated
    theses (ai_cake / nvda_halo / construction / general); ad-hoc
    syntheses live in their own DB so they don't pollute the curated
    list. Both share the same shape (Name, Slug, Universe, Anchors)
    plus an Origin field flagging "topic" vs "ticker" mode.

    Caller resolves `db_id` from `NOTION_DB_ADHOC_THESES` env var; this
    function expects it to already be present on disk. Bootstrap creation
    of the DB is intentionally manual — `python -m scripts.bootstrap_notion`
    will be extended in a future commit; for now the user creates it via
    the Notion UI when they're ready to use the mirror.
    """
    client = _get_client()
    if client is None or not db_id:
        return None
    universe = list(getattr(thesis, "universe", None) or [])
    anchors = list(getattr(thesis, "anchor_tickers", None) or [])
    name = getattr(thesis, "name", slug) or slug
    summary = getattr(thesis, "summary", "") or ""

    properties: dict[str, Any] = {
        "Name": {"title": [{"type": "text", "text": {"content": name}}]},
        "Slug": {
            "rich_text": [{"type": "text", "text": {"content": slug}}]
        },
        "Universe": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": ", ".join(universe)[:1900]},
                }
            ]
        },
        "Anchors": {
            "rich_text": [{"type": "text", "text": {"content": ", ".join(anchors)}}]
        },
        "Created": {"date": {"start": datetime.now(UTC).date().isoformat()}},
    }

    # Page body holds the summary so the user can browse the synthesis
    # rationale in Notion without round-tripping to the JSON on disk.
    children: list[dict] = []
    if summary:
        children.append(
            {
                "type": "paragraph",
                "paragraph": {"rich_text": _rich_text(summary)},
            }
        )

    try:
        page = client.pages.create(
            parent={"database_id": db_id},
            properties=properties,
            children=children if children else None,
        )
        return page.get("url")
    except Exception as e:
        logger.error(f"[notion] write_adhoc_thesis failed for {slug}: {e}")
        return None


def append_thesis_note(
    *,
    thesis_slug: str,
    text: str,
    ticker: str | None = None,
) -> bool:
    """Append a note to a thesis's Notion page.

    The note is appended as a single paragraph block at the bottom of the
    thesis row's page. The body is prefixed with `[YYYY-MM-DD] TICKER:`
    so when Triage reads the notes back via `read_thesis_notes()`, it sees
    when the note was added and (optionally) which ticker it scoped to.

    Returns True on success, False on no-op (Notion not configured) or
    failure (Notion error). Caller should surface failure to the user but
    NOT crash the bot — notes are best-effort.

    Phase 1 Triage will read these notes verbatim and weight them when
    scoring material thresholds. Free-text notes outweigh static thesis
    JSON for the user's evolving view; that's the whole point of the
    feedback loop.
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_THESES")
    if client is None or not db_id:
        return False
    ds_id = _resolve_data_source_id(db_id)
    if not ds_id:
        return False
    try:
        # 1. Find the thesis row by slug.
        rows = client.data_sources.query(
            data_source_id=ds_id,
            filter={"property": "Slug", "rich_text": {"equals": thesis_slug}},
            page_size=1,
        )
        results = rows.get("results", [])
        if not results:
            logger.warning(
                f"[notion] thesis row for slug={thesis_slug!r} not found — "
                f"can't append note. Has bootstrap_notion seeded the row?"
            )
            return False
        page_id = results[0]["id"]

        # 2. Append a paragraph block with the timestamped note.
        today = datetime.now(UTC).date().isoformat()
        prefix = (
            f"[{today}] {ticker.upper()}: " if ticker else f"[{today}] "
        )
        body = prefix + text.strip()
        client.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": _rich_text(body)},
                }
            ],
        )
        return True
    except Exception as e:
        logger.error(
            f"[notion] append_thesis_note failed for slug={thesis_slug}: {e}"
        )
        return False


# --- read_watchlist --------------------------------------------------------


def read_watchlist(thesis_name: str | None = None) -> list[dict]:
    """Return active watchlist rows as a list of dicts. Filters by thesis
    when a name is given. Phase 1 Triage reads this to pick up standing
    'watch this' rules across drill-ins.

    Each dict: `{item, ticker, thesis, agent, run_id, created}`.
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_WATCHLIST")
    if client is None or not db_id:
        return []
    ds_id = _resolve_data_source_id(db_id)
    if not ds_id:
        return []
    try:
        query: dict[str, Any] = {"data_source_id": ds_id, "page_size": 50}
        if thesis_name:
            query["filter"] = {
                "property": "Thesis",
                "rich_text": {"equals": thesis_name},
            }
        results = client.data_sources.query(**query)
        out: list[dict] = []
        for row in results.get("results", []):
            props = row.get("properties") or {}
            out.append({
                "item": _extract_title(props.get("Item")),
                "ticker": _extract_text(props.get("Ticker")),
                "thesis": _extract_text(props.get("Thesis")),
                "agent": _extract_select(props.get("Agent")),
                "run_id": _extract_text(props.get("Run ID")),
                "created": _extract_date(props.get("Created")),
                "id": row.get("id"),
            })
        return out
    except Exception as e:
        logger.warning(f"[notion] read_watchlist failed: {e}")
        return []


def write_watchlist_items(
    *,
    items: list[str],
    ticker: str,
    thesis_name: str,
    run_id: str | None = None,
) -> int:
    """Persist a list of watchlist strings (with their `(agent)` suffix) as
    rows in the Watchlist DB. Idempotent on (item, ticker, thesis) — if a
    row already exists for the same item + ticker + thesis, we skip it so
    repeated drill-ins don't pile up duplicates.

    Returns the count of rows actually inserted.
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_WATCHLIST")
    if client is None or not db_id or not items:
        return 0
    ds_id = _resolve_data_source_id(db_id)

    # Pull existing items for this thesis × ticker once, dedupe by exact
    # match — the Watchlist DB shouldn't grow without bound.
    existing_strings: set[str] = set()
    if ds_id:
        try:
            existing = client.data_sources.query(
                data_source_id=ds_id,
                filter={
                    "and": [
                        {"property": "Ticker", "rich_text": {"equals": ticker.upper()}},
                        {"property": "Thesis", "rich_text": {"equals": thesis_name}},
                    ]
                },
                page_size=100,
            )
            for row in existing.get("results", []):
                title = _extract_title((row.get("properties") or {}).get("Item"))
                if title:
                    existing_strings.add(title.strip())
        except Exception as e:
            logger.warning(f"[notion] watchlist dedupe-query failed: {e}")

    inserted = 0
    today = datetime.now(UTC).date().isoformat()
    agent_suffix_re = re.compile(r"\(([^)]+)\)\s*$")
    for raw_item in items:
        item = (raw_item or "").strip()
        if not item or item in existing_strings:
            continue
        m = agent_suffix_re.search(item)
        agent_value = m.group(1).strip().lower() if m else "synthesis"
        properties: dict[str, Any] = {
            "Item": {"title": [{"type": "text", "text": {"content": item}}]},
            "Ticker": {
                "rich_text": [{"type": "text", "text": {"content": ticker.upper()}}]
            },
            "Thesis": {
                "rich_text": [{"type": "text", "text": {"content": thesis_name}}]
            },
            "Agent": {"select": {"name": agent_value}},
            "Created": {"date": {"start": today}},
        }
        if run_id:
            properties["Run ID"] = {
                "rich_text": [{"type": "text", "text": {"content": str(run_id)}}]
            }
        try:
            client.pages.create(parent={"database_id": db_id}, properties=properties)
            inserted += 1
        except Exception as e:
            logger.warning(f"[notion] watchlist insert failed for {item!r}: {e}")
    return inserted


# --- write_alert / update_alert_status (Phase 1 Triage entry points) ------


def write_alert(
    *,
    ticker: str,
    thesis_name: str,
    severity: int,
    signal: str,
    evidence_url: str | None = None,
    run_id: str | None = None,
) -> tuple[str, str] | None:
    """Insert a new alert row in 'pending' status. Returns (alert_id, page_url)
    on success, None on no-op / failure.

    Wired by Phase 1 Triage (Step 11) — the function exists in Step 9 so
    the surface is testable up front and the bootstrap creates the DB now.
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_ALERTS")
    if client is None or not db_id:
        return None
    properties: dict[str, Any] = {
        "Signal": {"title": [{"type": "text", "text": {"content": signal}}]},
        "Ticker": {"rich_text": [{"type": "text", "text": {"content": ticker.upper()}}]},
        "Thesis": {"rich_text": [{"type": "text", "text": {"content": thesis_name}}]},
        "Severity": {"number": int(severity)},
        "Status": {"select": {"name": "pending"}},
        "Created": {"date": {"start": datetime.now(UTC).isoformat()}},
    }
    if evidence_url:
        properties["Evidence URL"] = {"url": evidence_url}
    if run_id:
        properties["Run ID"] = {
            "rich_text": [{"type": "text", "text": {"content": str(run_id)}}]
        }
    try:
        page = client.pages.create(parent={"database_id": db_id}, properties=properties)
        return page.get("id"), page.get("url")
    except Exception as e:
        logger.error(f"[notion] write_alert failed: {e}")
        return None


def update_alert_status(alert_id: str, status: str) -> bool:
    """Flip an alert's status field. Phase 1 Triage / dashboard / Telegram
    use this. Status ∈ {pending, acked, dismissed, actioned}."""
    if status not in ("pending", "acked", "dismissed", "actioned"):
        raise ValueError(f"invalid status {status!r}")
    client = _get_client()
    if client is None:
        return False
    try:
        client.pages.update(
            page_id=alert_id,
            properties={"Status": {"select": {"name": status}}},
        )
        return True
    except Exception as e:
        logger.error(f"[notion] update_alert_status failed: {e}")
        return False


def read_recent_reports(limit: int = 10) -> list[dict]:
    """Return the most-recent Reports-DB rows as dicts. Used by Mission
    Control's Notion panel to surface the cross-session report history.

    Each dict: `{title, ticker, thesis, confidence, date, url}`.
    """
    client = _get_client()
    db_id = _db_id("NOTION_DB_REPORTS")
    if client is None or not db_id:
        return []
    ds_id = _resolve_data_source_id(db_id)
    if not ds_id:
        return []
    try:
        results = client.data_sources.query(
            data_source_id=ds_id,
            sorts=[{"property": "Date", "direction": "descending"}],
            page_size=limit,
        )
        out: list[dict] = []
        for row in results.get("results", []):
            props = row.get("properties") or {}
            out.append({
                "title": _extract_title(props.get("Name")),
                "ticker": _extract_text(props.get("Ticker")),
                "thesis": _extract_text(props.get("Thesis")),
                "confidence": _extract_select(props.get("Confidence")),
                "date": _extract_date(props.get("Date")),
                "url": row.get("url"),
            })
        return out
    except Exception as e:
        logger.warning(f"[notion] read_recent_reports failed: {e}")
        return []


# --- Property extraction helpers -------------------------------------------


def _extract_title(prop: dict | None) -> str:
    if not prop:
        return ""
    parts = (prop.get("title") or []) if isinstance(prop, dict) else []
    return "".join(p.get("plain_text") or p.get("text", {}).get("content") or "" for p in parts)


def _extract_text(prop: dict | None) -> str:
    if not prop:
        return ""
    parts = (prop.get("rich_text") or []) if isinstance(prop, dict) else []
    return "".join(p.get("plain_text") or p.get("text", {}).get("content") or "" for p in parts)


def _extract_select(prop: dict | None) -> str:
    if not prop or not isinstance(prop, dict):
        return ""
    sel = prop.get("select")
    return sel.get("name", "") if isinstance(sel, dict) else ""


def _extract_date(prop: dict | None) -> str:
    if not prop or not isinstance(prop, dict):
        return ""
    date = prop.get("date")
    return date.get("start", "") if isinstance(date, dict) else ""
