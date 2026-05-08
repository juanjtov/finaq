"""Background-thread runner for FINAQ drill-ins.

Streamlit re-executes the page script on every interaction (button click,
tab switch, page navigation). If the LangGraph drill-in runs synchronously
inside that script, switching tabs / pages KILLS the in-flight `asyncio.run`.

This module spawns the drill-in in a daemon thread the first time the user
clicks 🔍 Run drill-in. The thread:
  - Has its own `current_run_id` ContextVar (asyncio.run inherits it).
  - Writes graph_runs + node_runs telemetry as usual via state.db.
  - Saves the final FinaqState to `data_cache/demos/{TICKER}__{slug}__{run_id[:8]}.json`.

The dashboard polls `is_running(ticker, slug)` on each rerun. While a run is
active it renders a "🏃 Running…" panel and re-runs itself every 2s. When the
run completes, the cached file is on disk and the standard load path picks
it up.

Why a module-level dict (not st.session_state):
  - session_state is per-tab — a user navigating from Dashboard to
    Architecture would lose the "in-progress" handle.
  - Module globals live in the Streamlit server process, shared across
    every tab in the browser. They die on server restart, which is fine
    (state.db's graph_runs row would say 'running' until cleaned up — see
    `cleanup_stale_runs()`).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

from utils import logger

# --- Module singleton -------------------------------------------------------

_lock = threading.Lock()
_active_runs: dict[tuple[str, str], dict[str, Any]] = {}
# key: (ticker_upper, thesis_slug)
# value: {"thread": Thread, "run_id": str|None, "started_at": float, "error": str|None}


# --- Public API -------------------------------------------------------------


def kick_off_drill(ticker: str, thesis_slug: str) -> bool:
    """Start a drill-in in a background daemon thread.

    Returns True if a new run was started, False if there's already one
    in flight for the same ticker × thesis (caller should just show the
    running panel and let it complete).
    """
    key = (ticker.upper(), thesis_slug)
    with _lock:
        existing = _active_runs.get(key)
        if existing and existing["thread"].is_alive():
            return False
        record = {
            "thread": None,  # filled below
            "run_id": None,  # set by the worker once invoke_with_telemetry stamps state
            "started_at": time.time(),
            "error": None,
        }
        _active_runs[key] = record
        thread = threading.Thread(
            target=_worker,
            args=(ticker, thesis_slug, record),
            daemon=True,
            name=f"finaq-drill-{ticker}-{thesis_slug}",
        )
        record["thread"] = thread
        thread.start()
    return True


def is_running(ticker: str, thesis_slug: str) -> bool:
    """True if a drill-in for this ticker × thesis is currently in flight."""
    key = (ticker.upper(), thesis_slug)
    rec = _active_runs.get(key)
    return rec is not None and rec["thread"] is not None and rec["thread"].is_alive()


def get_run_status(ticker: str, thesis_slug: str) -> dict | None:
    """Return the full record for the most recent run of this ticker × thesis,
    or None if no run has been kicked off this session."""
    return _active_runs.get((ticker.upper(), thesis_slug))


def elapsed_seconds(ticker: str, thesis_slug: str) -> float | None:
    """Wall-clock seconds since the run kicked off. None if no record."""
    rec = get_run_status(ticker, thesis_slug)
    if rec is None:
        return None
    return time.time() - rec["started_at"]


def clear_record(ticker: str, thesis_slug: str) -> None:
    """Remove the record once the dashboard has consumed the result. Lets a
    subsequent click re-run cleanly."""
    key = (ticker.upper(), thesis_slug)
    with _lock:
        _active_runs.pop(key, None)


# --- Worker thread ---------------------------------------------------------


def _worker(ticker: str, thesis_slug: str, record: dict[str, Any]) -> None:
    """Run the LangGraph drill-in inside this thread.

    ContextVars are NOT inherited across threads by default; that's fine —
    `invoke_with_telemetry` sets `current_run_id` inside its OWN coroutine,
    and that coroutine runs inside this thread's `asyncio.run`. So all
    `_safe_node` writes still see the right run_id.

    After the drill-in completes, optionally fire the per-drill-in RAG
    eval sidecar (utils.live_eval.evaluate_filings_retrieval). Gated by
    `EVAL_LIVE_DRILL_INS=true` env var so the cost is opt-in.
    """
    try:
        from agents import build_graph, invoke_with_telemetry

        thesis_path = Path("theses") / f"{thesis_slug}.json"
        thesis = json.loads(thesis_path.read_text())
        # Step 11.20 — stamp slug on the dict so `invoke_with_telemetry`
        # writes graph_runs.thesis = slug (canonical), not the human name.
        # Without this, the CIO planner's cooldown gate can't join its
        # `cio_actions.thesis` (slug) to graph_runs.
        thesis.setdefault("slug", thesis_slug)
        graph = build_graph()

        async def _run() -> dict[str, Any]:
            return await invoke_with_telemetry(
                graph, {"ticker": ticker.upper(), "thesis": thesis}
            )

        final = asyncio.run(_run())
        record["run_id"] = final.get("run_id")

        # Sidecar 1: per-drill-in RAG eval. Opt-in via EVAL_LIVE_DRILL_INS.
        _maybe_run_live_eval(final, thesis)

        # Sidecar 2: write report + watchlist to Notion. No-op when
        # NOTION_API_KEY isn't set. Best-effort — never blocks the drill-in.
        # Mutates `final` in place to add `notion_report_url` so downstream
        # consumers (Telegram /drill reply, Mission Control) can link out.
        _maybe_write_to_notion(final, thesis)

        # Persist the cached demo AFTER sidecars so the saved JSON contains
        # any URLs / sidecar additions. The dashboard load path picks it up.
        _save_run_to_demo_dir(ticker.upper(), thesis_slug, final)
    except Exception as e:
        logger.error(f"[runner] drill-in failed for {ticker} × {thesis_slug}: {e}")
        record["error"] = str(e)


def _maybe_write_to_notion(final: dict[str, Any], thesis: dict) -> None:
    """Persist the synthesis report + watchlist items to Notion. No-op when
    Notion isn't configured (NOTION_API_KEY unset). Errors logged but never
    propagate — the dashboard should never block on a Notion outage."""
    try:
        from data import notion as _notion
    except ImportError:
        return
    if not _notion.is_configured():
        return
    report_md = final.get("report") or ""
    if not report_md.strip():
        return
    ticker = final.get("ticker") or "?"
    thesis_name = (
        thesis.get("name") if isinstance(thesis, dict) else str(thesis or "?")
    )
    fund = final.get("fundamentals") or {}
    kpis = fund.get("kpis") or {}
    mc = final.get("monte_carlo") or {}
    p50 = (mc.get("dcf") or {}).get("p50") if isinstance(mc.get("dcf"), dict) else None
    try:
        url = _notion.write_report(
            ticker=ticker,
            thesis_name=thesis_name,
            markdown=report_md,
            confidence=final.get("synthesis_confidence"),
            p50=p50,
            current_price=kpis.get("current_price"),
            run_id=final.get("run_id"),
        )
        if url:
            # Stash on the state so Telegram /drill replies and Mission
            # Control can link out without re-querying Notion.
            final["notion_report_url"] = url
            logger.info(f"[runner] notion report persisted: {url}")
    except Exception as e:
        logger.warning(f"[runner] notion report write failed: {e}")

    # Persist the watchlist items as separate rows so Phase 1 Triage can
    # read them back and turn them into monitoring rules.
    watchlist = final.get("watchlist") or []
    if watchlist:
        try:
            inserted = _notion.write_watchlist_items(
                items=watchlist,
                ticker=ticker,
                thesis_name=thesis_name,
                run_id=final.get("run_id"),
            )
            if inserted:
                logger.info(f"[runner] notion watchlist: +{inserted} item(s)")
        except Exception as e:
            logger.warning(f"[runner] notion watchlist write failed: {e}")


def _maybe_run_live_eval(final: dict[str, Any], thesis: dict) -> None:
    """Fire utils.live_eval.evaluate_filings_retrieval if enabled. Reads the
    `_retrieval_audit` field that `agents.filings.run` stashes on its
    payload — that's how we grade WITHOUT re-running RAG."""
    try:
        from utils import live_eval as _live_eval
    except ImportError:
        return
    if not _live_eval.is_enabled():
        return
    filings = final.get("filings") or {}
    audit = filings.get("_retrieval_audit") or []
    if not audit:
        return
    # Reconstruct (subquery_dict, chunks) tuples expected by evaluate_filings_retrieval.
    subqueries_with_chunks = [
        (
            {"label": entry.get("label"), "question": entry.get("question")},
            entry.get("chunks") or [],
        )
        for entry in audit
    ]
    try:
        _live_eval.evaluate_filings_retrieval(
            run_id=final.get("run_id"),
            ticker=final.get("ticker", "?"),
            thesis_name=(thesis.get("name") if isinstance(thesis, dict) else "?"),
            subqueries_with_chunks=subqueries_with_chunks,
        )
        logger.info(
            f"[runner] live RAG eval persisted for run_id={final.get('run_id', '?')}"
        )
    except Exception as e:
        logger.warning(f"[runner] live RAG eval failed: {e}")


def _save_run_to_demo_dir(ticker: str, thesis_slug: str, final_state: dict) -> Path:
    """Write the final state to data_cache/demos/{TICKER}__{slug}__{run_id[:8]}.json.
    Mirrors `ui/app.py:_save_demo` (kept inline here to avoid a circular
    Streamlit import — `ui/app.py` is the dashboard entrypoint and importing
    from it would re-run page setup)."""
    demo_dir = Path("data_cache/demos")
    demo_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in final_state.items() if k != "messages"}
    run_id = final_state.get("run_id")
    suffix = f"__{run_id[:8]}" if run_id else ""
    path = demo_dir / f"{ticker}__{thesis_slug}{suffix}.json"
    path.write_text(json.dumps(payload, default=str, indent=2))
    return path
