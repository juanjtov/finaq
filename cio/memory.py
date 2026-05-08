"""CIO domain layer over `data.state` (state.db) + `data.notion`.

Provides the CIO planner with a small, intentional surface for the only
queries it needs:

  - Cooldown:       has this (ticker, thesis) been drilled in the last N hours?
  - Recent actions: what did the CIO last decide for this pair?
  - User notes:     what has the user told us via Notion `/note`?
  - Last drill_run_id: the run_id the planner can hand back as `reuse_run_id`
                       when it picks `action=reuse`.

Why not call `data.state` and `data.notion` directly from the planner?
Because the planner shouldn't have to know SQL column names or the Notion
SDK's failure modes. This module hides both.

All Notion calls are best-effort: if Notion is unconfigured or the page
doesn't exist, return empty data. The planner treats "no notes" the same
as "user has nothing to say about this thesis right now".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from data import state as state_db
from utils import logger

# Cooldown defaults — locked in the v1 plan. Override per-call if a future
# `/cio` flag wants to ignore the cooldown.
DEFAULT_COOLDOWN_HOURS = 48


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def cooldown_status(
    ticker: str,
    thesis: str | None = None,
    *,
    cooldown_hours: int = DEFAULT_COOLDOWN_HOURS,
) -> dict:
    """Return cooldown status for a (ticker, thesis) pair.

    Output:
      {
        "active":               bool,    # True if last drill < cooldown_hours ago
        "last_drill_age_hours": float | None,
        "last_drill_run_id":    str   | None,  # for `reuse_run_id` routing
        "last_drill_at":        str   | None,  # ISO ts of last drill
      }

    The planner uses this as a soft gate: cooldown=active biases toward
    reuse / dismiss; cooldown=expired biases toward fresh drill.
    """
    last = state_db.last_drill_for(ticker, thesis)
    if not last:
        return {
            "active": False,
            "last_drill_age_hours": None,
            "last_drill_run_id": None,
            "last_drill_at": None,
        }
    ended = _parse_iso(last.get("ended_at")) or _parse_iso(last.get("started_at"))
    if not ended:
        return {
            "active": False,
            "last_drill_age_hours": None,
            "last_drill_run_id": last.get("run_id"),
            "last_drill_at": None,
        }
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - ended).total_seconds() / 3600.0
    return {
        "active": age < cooldown_hours,
        "last_drill_age_hours": round(age, 2),
        "last_drill_run_id": last.get("run_id"),
        "last_drill_at": ended.isoformat(),
    }


def recent_cio_actions(
    ticker: str,
    thesis: str | None = None,
    *,
    limit: int = 5,
) -> list[dict]:
    """Last N CIO actions for the pair, most-recent-first. The planner
    uses these to avoid yo-yo decisions ("we just dismissed this 4h ago,
    why drill now?")."""
    return state_db.recent_cio_actions(limit=limit, ticker=ticker, thesis=thesis)


def thesis_notes(thesis: str) -> str:
    """User's Notion notes for a thesis, as plain text. Empty string when
    Notion is unconfigured or the slug doesn't match. Soft-fail: never
    raises — Notion outage is not a CIO blocker.

    The planner folds this into its prompt so user-asserted preferences
    ("trim 20% if Q3 misses $42B") shape decisions without us having to
    encode them as material_thresholds in the JSON.
    """
    try:
        from data import notion as _notion

        if not _notion.is_configured():
            return ""
        return _notion.read_thesis_notes(thesis) or ""
    except Exception as e:
        logger.warning(f"[cio.memory] thesis_notes({thesis!r}) failed: {e}")
        return ""


def dismissals_in_window(
    ticker: str,
    thesis: str | None = None,
    *,
    window_days: int = 7,
) -> list[dict]:
    """All `dismiss` actions for the pair in the last `window_days` days.

    The planner uses this as an anti-yo-yo signal — if we dismissed the
    same pair 3 times this week, we don't need to LLM-decide a 4th time;
    we can shortcut to dismiss without burning tokens.
    """
    actions = recent_cio_actions(ticker, thesis, limit=20)
    cutoff = datetime.now(UTC) - timedelta(days=window_days)
    out: list[dict] = []
    for a in actions:
        if a.get("action") != "dismiss":
            continue
        ts = _parse_iso(a.get("ts"))
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts and ts >= cutoff:
            out.append(a)
    return out
