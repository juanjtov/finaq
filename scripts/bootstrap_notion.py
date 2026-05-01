"""One-time setup for FINAQ ↔ Notion (Step 9).

Creates four databases under a parent page in your Notion workspace:
  - Theses    : your free-text annotations per thesis (system reads these)
  - Reports   : every drill-in's synthesis report lands here as a row
  - Alerts    : Phase 1 Triage emits material alerts, you ack/dismiss
  - Watchlist : Synthesis-emitted "track this next quarter" items

Idempotent — re-running detects existing databases (matched by Title) under
the parent page and reports their IDs without recreating.

Prereqs (do these in your browser first):
  1. Create an internal integration at https://www.notion.so/my-integrations
  2. Copy the integration token → `.env` as NOTION_API_KEY=ntn_...
  3. Create or pick a parent page in your workspace.
  4. On that page → top-right "..." → Connections → add your integration.
  5. Copy the parent page ID (32-char hex, last segment of the URL).
  6. `.env`: NOTION_PARENT_PAGE_ID=<the hex>

Usage:
  python -m scripts.bootstrap_notion              # create the 4 DBs
  python -m scripts.bootstrap_notion --list-only  # show what's already there

After it runs, paste the printed `NOTION_DB_*` lines into `.env` and restart
Streamlit. Subsequent drill-ins will write reports to your Reports DB.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

from utils import logger

load_dotenv()


# --- Database schemas ------------------------------------------------------
#
# Each schema is the `properties` arg of `databases.create`. Notion property
# types:
#   title           — primary identifier (always required, exactly one per DB)
#   rich_text       — multi-line text, can hold inline annotations
#   number          — numeric (with optional format)
#   date            — ISO date or datetime
#   select / status — controlled vocab
#   url             — clickable link
#
# The keys here are the user-facing column names; the FINAQ code in
# `data/notion.py` references them by these exact strings, so don't rename
# without updating the consumer at the same time.


_THESES_SCHEMA: dict[str, Any] = {
    "Name": {"title": {}},
    "Slug": {"rich_text": {}},  # FINAQ reads via this — must match thesis JSON filename
    "Universe": {"rich_text": {}},  # comma-separated tickers, for at-a-glance browsing
    "Anchors": {"rich_text": {}},
    "Last Drill-in": {"date": {}},
}

_REPORTS_SCHEMA: dict[str, Any] = {
    "Name": {"title": {}},
    "Ticker": {"rich_text": {}},
    "Thesis": {"rich_text": {}},
    "Date": {"date": {}},
    "Confidence": {
        "select": {
            "options": [
                {"name": "low", "color": "gray"},
                {"name": "medium", "color": "yellow"},
                {"name": "high", "color": "green"},
            ]
        }
    },
    "DCF P50": {"number": {"format": "dollar"}},
    "Current Price": {"number": {"format": "dollar"}},
    "Run ID": {"rich_text": {}},
}

_ALERTS_SCHEMA: dict[str, Any] = {
    "Signal": {"title": {}},
    "Ticker": {"rich_text": {}},
    "Thesis": {"rich_text": {}},
    "Severity": {"number": {"format": "number"}},
    "Status": {
        "select": {
            "options": [
                {"name": "pending", "color": "yellow"},
                {"name": "acked", "color": "blue"},
                {"name": "dismissed", "color": "gray"},
                {"name": "actioned", "color": "green"},
            ]
        }
    },
    "Evidence URL": {"url": {}},
    "Run ID": {"rich_text": {}},
    "Created": {"date": {}},
}

_WATCHLIST_SCHEMA: dict[str, Any] = {
    "Item": {"title": {}},
    "Ticker": {"rich_text": {}},
    "Thesis": {"rich_text": {}},
    "Agent": {
        "select": {
            "options": [
                {"name": "fundamentals", "color": "brown"},
                {"name": "filings", "color": "green"},
                {"name": "news", "color": "orange"},
                {"name": "risk", "color": "red"},
                {"name": "synthesis", "color": "purple"},
            ]
        }
    },
    "Run ID": {"rich_text": {}},
    "Created": {"date": {}},
}

# Mapping `env_var → (display_title, schema, env_var_name)`. The keys are the
# stable internal names; the env-var-name is what users paste back into `.env`.
_DBS: list[tuple[str, dict[str, Any], str]] = [
    ("FINAQ — Theses", _THESES_SCHEMA, "NOTION_DB_THESES"),
    ("FINAQ — Reports", _REPORTS_SCHEMA, "NOTION_DB_REPORTS"),
    ("FINAQ — Alerts", _ALERTS_SCHEMA, "NOTION_DB_ALERTS"),
    ("FINAQ — Watchlist", _WATCHLIST_SCHEMA, "NOTION_DB_WATCHLIST"),
]


# --- Helpers ---------------------------------------------------------------


def _normalise_page_id(raw: str) -> str:
    """Notion page IDs are 32-char hex with optional dashes. Accept either."""
    return raw.replace("-", "").strip()


def _find_existing_db(client: Any, parent_id: str, title: str) -> str | None:
    """Return the database ID if a non-archived child of `parent_id` with the
    given title already exists. Idempotency for re-runs.

    Notion's blocks.children.list returns archived child_databases too, so we
    confirm via databases.retrieve before returning a match — an archived
    DB shouldn't satisfy "already exists".
    """
    try:
        children = client.blocks.children.list(block_id=parent_id, page_size=100)
    except Exception as e:
        logger.warning(f"[bootstrap] could not list parent children: {e}")
        return None
    for block in children.get("results", []):
        if block.get("type") != "child_database":
            continue
        cd = block.get("child_database") or {}
        if cd.get("title") != title:
            continue
        block_id = block.get("id")
        if not block_id:
            continue
        try:
            db = client.databases.retrieve(database_id=block_id)
        except Exception:
            continue
        if db.get("archived") or db.get("in_trash"):
            continue
        return block_id
    return None


def _create_db(
    client: Any, parent_id: str, title: str, schema: dict[str, Any]
) -> str:
    """Create a new database under `parent_id` with the given schema. Returns
    the new database ID.

    Notion API 2025-09-03 introduced "data sources": a database is a container
    that holds one or more data sources, and the schema (properties) lives on
    the data source. Passing `properties=` directly to `databases.create` is
    silently dropped under this version. The correct shape is to pass
    `initial_data_source={"properties": ...}` which creates the database with
    its default data source pre-populated.
    """
    response = client.databases.create(
        parent={"type": "page_id", "page_id": parent_id},
        title=[{"type": "text", "text": {"content": title}}],
        initial_data_source={"properties": schema},
    )
    return response["id"]


def _seed_thesis_rows(client: Any, db_id: str) -> int:
    """Pre-populate the Theses DB with one row per thesis JSON in /theses/.
    Each row carries the slug + universe + anchors from the JSON; the page
    body is empty so the user can fill it with their own annotations."""
    import json as _json
    from pathlib import Path

    theses_dir = Path("theses")
    if not theses_dir.exists():
        return 0
    # Resolve the database's default data_source_id once — Notion API 2025-09-03
    # routes queries through `data_sources.query`, not `databases.query`.
    try:
        db = client.databases.retrieve(database_id=db_id)
        data_source_id = (db.get("data_sources") or [{}])[0].get("id")
    except Exception as e:
        logger.warning(f"[bootstrap] could not resolve data_source_id: {e}")
        data_source_id = None
    inserted = 0
    for path in sorted(theses_dir.glob("*.json")):
        try:
            data = _json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"[bootstrap] could not read {path}: {e}")
            continue
        slug = path.stem
        # Skip if a row with this slug already exists
        if data_source_id:
            try:
                existing = client.data_sources.query(
                    data_source_id=data_source_id,
                    filter={"property": "Slug", "rich_text": {"equals": slug}},
                    page_size=1,
                )
                if existing.get("results"):
                    continue
            except Exception:
                pass
        try:
            client.pages.create(
                parent={"database_id": db_id},
                properties={
                    "Name": {
                        "title": [{"type": "text", "text": {"content": data.get("name", slug)}}]
                    },
                    "Slug": {"rich_text": [{"type": "text", "text": {"content": slug}}]},
                    "Universe": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": ", ".join(data.get("universe") or [])[:1900]},
                            }
                        ]
                    },
                    "Anchors": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": ", ".join(data.get("anchor_tickers") or [])},
                            }
                        ]
                    },
                },
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"[bootstrap] seed-row insert failed for {slug}: {e}")
    return inserted


# --- Main ------------------------------------------------------------------


def main(list_only: bool = False) -> int:
    api_key = os.environ.get("NOTION_API_KEY", "").strip()
    parent_id_raw = os.environ.get("NOTION_PARENT_PAGE_ID", "").strip()
    if not api_key:
        print("ERROR: NOTION_API_KEY not set in .env", file=sys.stderr)
        return 1
    if not parent_id_raw:
        print("ERROR: NOTION_PARENT_PAGE_ID not set in .env", file=sys.stderr)
        return 1
    parent_id = _normalise_page_id(parent_id_raw)

    try:
        from notion_client import Client
    except ImportError:
        print(
            "ERROR: notion-client not installed. Run `pip install notion-client`.",
            file=sys.stderr,
        )
        return 1
    client = Client(auth=api_key)

    # Sanity check: can we actually see the parent page?
    try:
        page = client.pages.retrieve(parent_id)
        page_title = ""
        for prop in (page.get("properties") or {}).values():
            if prop.get("type") == "title":
                title_parts = prop.get("title") or []
                page_title = "".join(p.get("plain_text", "") for p in title_parts)
                break
        print(f"✓ Connected to Notion. Parent page: {page_title or parent_id}")
    except Exception as e:
        print(
            f"ERROR: could not access parent page {parent_id}: {e}\n"
            "Did you share the page with your integration? "
            "(Page → ... → Connections → add the integration)",
            file=sys.stderr,
        )
        return 1

    print()
    env_lines: list[str] = []
    for title, schema, env_var in _DBS:
        existing = _find_existing_db(client, parent_id, title)
        if existing:
            db_id = existing
            print(f"  ✓ {title}: already exists ({db_id})")
        elif list_only:
            print(f"  ✗ {title}: NOT FOUND (run without --list-only to create)")
            continue
        else:
            try:
                db_id = _create_db(client, parent_id, title, schema)
                print(f"  + {title}: created ({db_id})")
            except Exception as e:
                print(f"  ✗ {title}: creation failed: {e}", file=sys.stderr)
                continue
        env_lines.append(f"{env_var}={db_id}")

    if not list_only and env_lines:
        # Seed the Theses DB with one row per local thesis JSON for free.
        theses_db_id = next(
            (line.split("=", 1)[1] for line in env_lines if line.startswith("NOTION_DB_THESES=")),
            None,
        )
        if theses_db_id:
            seeded = _seed_thesis_rows(client, theses_db_id)
            if seeded:
                print(f"  + Seeded Theses DB with {seeded} thesis row(s)")

    print()
    print("=" * 70)
    print("Paste these lines into your .env (replacing any existing NOTION_DB_*):")
    print("=" * 70)
    for line in env_lines:
        print(line)
    print("=" * 70)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print existing DB IDs; don't create missing ones.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(list_only=args.list_only))
