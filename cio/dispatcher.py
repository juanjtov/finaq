"""CIO dispatcher — CLI entrypoint for the launchd / systemd timer.

Modes:
  - `auto`        — freshness check; runs catchup if last_successful is >8h old,
                    otherwise heartbeat. Used by the scheduled timer + RunAtLoad=true.
  - `heartbeat`   — force heartbeat regardless of freshness.
  - `catchup`     — force catchup (for manual recovery after a long outage).
  - `on_demand`   — `--ticker NVDA [--thesis ai_cake]`; the Telegram /cio command
                    reaches this path indirectly (it imports `cio.cio` directly,
                    not the CLI — but the same primitives drive both).

Exit codes:
  0 — completed successfully
  1 — completed with the cycle marked failed (cio_runs.status = 'failed')
  2 — argparse / config error before the cycle ever started

Catch-up rationale:
  The Mac is on PT and the user might close the lid (laptop) at the time
  the heartbeat is supposed to fire. launchd's `RunAtLoad=true` on the plist
  means we re-fire on user-login / wake. The dispatcher then checks
  `last_successful_cio_run_at` — if more than `CATCHUP_THRESHOLD_HOURS`
  have passed, we run a CATCHUP cycle (tagged differently from heartbeat
  in the cio_runs table) so the user knows this was a recovery cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Bootstrap so this can be run as `python -m cio.dispatcher`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cio import cio as cio_orchestrator
from cio import notify as cio_notify
from data import state as state_db
from utils import logger

CATCHUP_THRESHOLD_HOURS = 8.0
"""If `last_successful_cio_run_at` is older than this, `--mode auto`
fires a catchup instead of heartbeat. Locked in v1 — matches the gap
between the 8am ET and 4pm ET heartbeat slots, so a single missed slot
triggers exactly one recovery cycle (not N stacked retries)."""


def _hours_since(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (datetime.now(UTC) - dt).total_seconds() / 3600.0


def _resolve_auto_mode() -> str:
    """Return 'heartbeat' or 'catchup' based on freshness check.

    First-ever run (no successful cycle on disk yet) returns `catchup` so
    the boot path produces a baseline entry rather than waiting for the
    next scheduled tick.
    """
    last_iso = state_db.last_successful_cio_run_at()
    age_h = _hours_since(last_iso)
    if age_h is None:
        logger.info("[cio.dispatcher] no prior successful cycle; choosing CATCHUP")
        return "catchup"
    if age_h > CATCHUP_THRESHOLD_HOURS:
        logger.info(
            f"[cio.dispatcher] last successful cycle was {age_h:.1f}h ago "
            f"(>{CATCHUP_THRESHOLD_HOURS}h); choosing CATCHUP"
        )
        return "catchup"
    logger.info(
        f"[cio.dispatcher] last successful cycle was {age_h:.1f}h ago "
        f"(<{CATCHUP_THRESHOLD_HOURS}h); choosing HEARTBEAT"
    )
    return "heartbeat"


async def _run(mode: str, *, ticker: str | None, thesis: str | None) -> int:
    """Execute one cycle. Returns the process exit code.

    Always sends the Telegram + Notion notification on success.
    """
    if mode == "auto":
        mode = _resolve_auto_mode()

    if mode == "heartbeat":
        plan, summary = await cio_orchestrator.run_heartbeat()
    elif mode == "catchup":
        plan, summary = await cio_orchestrator.run_catchup()
    elif mode == "on_demand":
        if not ticker:
            logger.error("[cio.dispatcher] --mode on_demand requires --ticker")
            return 2
        plan, summary = await cio_orchestrator.run_on_demand(ticker, thesis)
    else:
        logger.error(f"[cio.dispatcher] unknown mode {mode!r}")
        return 2

    # Duration from the cio_runs row that was just closed.
    runs = state_db.recent_cio_runs(limit=1)
    duration_s = float((runs[0].get("duration_s") if runs else 0.0) or 0.0)

    notify_result = cio_notify.notify_cycle(
        plan, trigger=mode, duration_s=duration_s, summary_text=summary,
    )
    logger.info(
        f"[cio.dispatcher] {mode} done: "
        f"{plan.n_drilled} drilled, {plan.n_reused} reused, "
        f"{plan.n_dismissed} dismissed; "
        f"telegram_sent={notify_result['telegram_sent']}, "
        f"notion={'yes' if notify_result['notion_url'] else 'no'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "heartbeat", "catchup", "on_demand"),
        default="auto",
        help="auto = freshness check; heartbeat / catchup force the mode; "
        "on_demand requires --ticker.",
    )
    parser.add_argument("--ticker", help="ticker (required for on_demand)")
    parser.add_argument("--thesis", help="thesis slug (optional, for on_demand)")
    args = parser.parse_args(argv)

    try:
        return asyncio.run(
            _run(args.mode, ticker=args.ticker, thesis=args.thesis)
        )
    except KeyboardInterrupt:
        logger.warning("[cio.dispatcher] interrupted")
        return 1
    except Exception as e:
        # Cycle-level fatal that didn't get caught inside _run_cycle.
        logger.error(f"[cio.dispatcher] fatal: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
