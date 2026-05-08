"""Thesis lifecycle primitives — promote / demote / archive.

The dashboard sidebar and `_list_thesis_slugs` glob `theses/*.json` and treat
any slug starting with `adhoc_` as an ad-hoc thesis (auto-generated from
`/analyze`). Curated theses (`ai_cake`, `construction`, `nvda_halo`,
`general`) drive the always-on CIO heartbeat sweep — adhoc theses don't.

Lifecycle:
  - `promote_thesis(adhoc_slug)` — adhoc → curated. Strips the `adhoc_`
    prefix and renames in-place. If a curated thesis already exists at the
    new name, it is archived first (never destroyed).
  - `demote_thesis(slug)` — curated → archive. Moves the curated JSON to
    `theses/archive/{ts}__{slug}.json` so it stops appearing on the
    dashboard / in CIO sweeps but the file is still recoverable.
  - `archive_thesis(slug)` — any → archive. Used internally by promote()
    when overwriting a pre-existing curated thesis, and by the admin
    page's "Archive" button. Never renames the slug.

All three return `tuple[bool, str]` so callers (Telegram handlers, the
Streamlit admin page) can render a one-line result message.

Schema validation: promote() runs the source through Pydantic before
moving anything — a malformed adhoc thesis must not enter the curated
set. demote() and archive() do not validate, so a malformed thesis can
still be moved out of the active set.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from utils import logger
from utils.schemas import Thesis

THESES_DIR = Path("theses")
ARCHIVE_DIR = THESES_DIR / "archive"
ADHOC_PREFIX = "adhoc_"


def _archive_timestamp() -> str:
    """Return an archive-filename timestamp. Indirected so tests can freeze it."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _archive_filename(slug: str) -> Path:
    """Build a unique archive path. Format: `{ts}__{slug}.json`.

    Collision handling: if a file already exists at the timestamped
    name (two archives of the same slug within one second), append
    `_1`, `_2`, etc. until we find a free name.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = _archive_timestamp()
    base = ARCHIVE_DIR / f"{ts}__{slug}.json"
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = ARCHIVE_DIR / f"{ts}__{slug}_{n}.json"
        if not candidate.exists():
            return candidate
        n += 1


def archive_thesis(slug: str) -> tuple[bool, str]:
    """Move `theses/{slug}.json` to `theses/archive/{ts}__{slug}.json`.

    Does NOT validate the file (a malformed thesis can still be archived)
    and does NOT rename the slug. Returns (ok, message).
    """
    src = THESES_DIR / f"{slug}.json"
    if not src.exists():
        return False, f"thesis {slug!r} not found at {src}"
    dst = _archive_filename(slug)
    try:
        shutil.move(str(src), str(dst))
    except OSError as e:
        return False, f"archive failed: {e}"
    logger.info(f"[theses] archived {slug} → {dst.name}")
    return True, f"archived → {dst.name}"


def promote_thesis(slug: str) -> tuple[bool, str]:
    """Promote an ad-hoc thesis to curated.

    `slug` must include the `adhoc_` prefix. The new curated slug is
    `slug[len(ADHOC_PREFIX):]`. If a curated thesis already exists at
    that name, it is archived first.

    Schema-validates the adhoc thesis before any move — a malformed
    file is rejected, so the curated set stays clean.
    """
    if not slug.startswith(ADHOC_PREFIX):
        return False, f"{slug!r} is not an adhoc slug (expected `adhoc_*`)"
    src = THESES_DIR / f"{slug}.json"
    if not src.exists():
        return False, f"adhoc thesis {slug!r} not found at {src}"
    new_slug = slug[len(ADHOC_PREFIX) :]
    if not new_slug:
        return False, f"slug {slug!r} would promote to an empty curated name"
    dst = THESES_DIR / f"{new_slug}.json"

    try:
        Thesis.model_validate_json(src.read_text())
    except (ValidationError, json.JSONDecodeError, OSError) as e:
        return False, f"adhoc thesis {slug!r} failed schema validation: {e}"

    if dst.exists():
        ok, msg = archive_thesis(new_slug)
        if not ok:
            return False, f"could not archive existing curated {new_slug!r}: {msg}"

    try:
        shutil.move(str(src), str(dst))
    except OSError as e:
        return False, f"promote rename failed: {e}"
    logger.info(f"[theses] promoted {slug} → {new_slug}")
    return True, f"promoted {slug} → {new_slug}"


def demote_thesis(slug: str) -> tuple[bool, str]:
    """Demote a curated thesis to the archive.

    Refuses adhoc slugs — those should be archived directly via
    `archive_thesis(slug)` since "demoting" an adhoc doesn't really
    mean anything (it was never curated).
    """
    if slug.startswith(ADHOC_PREFIX):
        return False, f"{slug!r} is an adhoc slug — use archive_thesis() instead"
    src = THESES_DIR / f"{slug}.json"
    if not src.exists():
        return False, f"curated thesis {slug!r} not found at {src}"
    return archive_thesis(slug)
