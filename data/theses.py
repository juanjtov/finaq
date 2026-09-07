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
from collections.abc import Iterable
from datetime import UTC, date, datetime
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
    # Promotion is a human judgement that the thesis is right — that is a
    # review. Soft-fail: the promote already succeeded.
    ok, msg = mark_reviewed(new_slug)
    if not ok:
        logger.warning(f"[theses] promoted {new_slug} but could not stamp last_reviewed: {msg}")
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


# --- Review age ------------------------------------------------------------
# "Is the thesis itself still what I believe?" is a human judgement the
# system can't make, but it can nag. `last_reviewed` (ISO date in the JSON)
# records the last time the user confirmed it; the CIO summary and Theses
# Admin flag anything older than REVIEW_MAX_DAYS. (User decision 2026-09-07.)

REVIEW_MAX_DAYS = 90


def _thesis_path(slug: str, theses_dir: Path | None = None) -> Path:
    return (theses_dir or THESES_DIR) / f"{slug}.json"


def review_age_days(
    slug: str,
    *,
    theses_dir: Path | None = None,
    today: date | None = None,
) -> int | None:
    """Days since the thesis was last reviewed.

    Reads `last_reviewed` from the JSON; falls back to the file's mtime
    (its last edit) when the field is absent or malformed, so theses
    written before the field existed still get an honest age. None when
    the file is missing or unparseable.
    """
    path = _thesis_path(slug, theses_dir)
    if not path.exists():
        return None
    today = today or datetime.now(UTC).date()
    reviewed: date | None = None
    try:
        raw = json.loads(path.read_text()).get("last_reviewed")
        if raw:
            reviewed = date.fromisoformat(str(raw)[:10])
    except (ValueError, TypeError, json.JSONDecodeError, OSError):
        reviewed = None
    if reviewed is None:
        try:
            reviewed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
        except OSError:
            return None
    return max(0, (today - reviewed).days)


def mark_reviewed(
    slug: str,
    *,
    theses_dir: Path | None = None,
    today: date | None = None,
) -> tuple[bool, str]:
    """Stamp `last_reviewed = today` into `theses/{slug}.json`, preserving
    every other key and the 2-space layout. Works for curated and adhoc."""
    path = _thesis_path(slug, theses_dir)
    if not path.exists():
        return False, f"thesis {slug!r} not found at {path}"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return False, f"could not read {slug!r}: {e}"
    stamp = (today or datetime.now(UTC).date()).isoformat()
    data["last_reviewed"] = stamp
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    except OSError as e:
        return False, f"could not write {slug!r}: {e}"
    logger.info(f"[theses] {slug} marked reviewed {stamp}")
    return True, f"{slug} marked reviewed {stamp}"


def overdue_theses(
    slugs: Iterable[str],
    *,
    max_days: int = REVIEW_MAX_DAYS,
    theses_dir: Path | None = None,
    today: date | None = None,
) -> list[dict]:
    """`[{"slug", "age_days"}]` for theses unreviewed for more than
    `max_days`, oldest first. Unknown ages (missing file) are skipped."""
    out: list[dict] = []
    for slug in sorted(set(slugs)):
        age = review_age_days(slug, theses_dir=theses_dir, today=today)
        if age is not None and age > max_days:
            out.append({"slug": slug, "age_days": age})
    out.sort(key=lambda r: -r["age_days"])
    return out
