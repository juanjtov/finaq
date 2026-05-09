"""One-ticker backtest runner: thesis → drill → score → persist.

The CLI in `scripts/backtest.py` walks the (ticker × as_of × horizons)
matrix and calls `run_backtest()` once per pair.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from utils import logger

RUNS_DIR = Path("data_cache/backtest/runs")


async def run_backtest(
    ticker: str,
    *,
    as_of_date: str,
    horizons: list[int],
    force_refresh_adhoc: bool = False,
) -> dict[str, Any]:
    """Run one backtest drill-in + score.

    Steps:
      1. Resolve thesis (curated lookup or date-pinned adhoc generation).
      2. Auto-ingest filings if the ticker has zero chunks in the corpus.
         Backtest mode's `data.edgar.download_filings(as_of=...)` won't
         fetch from EDGAR (post-as_of leakage risk), so we ingest
         WITHOUT as_of first if the corpus is empty, then the as_of
         filter narrows the result downstream.
      3. Build state with `as_of_date` set and invoke the graph.
      4. Score the result against realised prices at +N days for each horizon.
      5. Write `data_cache/backtest/runs/{TICKER}__{as_of}.json`.

    Returns the persisted dict.
    """
    from agents import build_graph, invoke_with_telemetry
    from backtest.scorer import score_run
    from backtest.thesis_resolver import resolve_thesis

    ticker = ticker.upper()
    started_at = time.perf_counter()

    # 1. Thesis resolution
    slug, thesis = await resolve_thesis(
        ticker, as_of_date=as_of_date, force_refresh_adhoc=force_refresh_adhoc,
    )
    logger.info(f"[backtest] {ticker} → thesis={slug!r} as_of={as_of_date}")

    # 2. Filings auto-ingest if corpus empty (production mode — captures
    #    everything; the per-call as_of filter on chroma.query handles the
    #    historical posture for retrieval).
    await _ensure_filings_ingested(ticker)

    # 3. Drill-in with as_of_date in state
    graph = build_graph()
    state = {
        "ticker": ticker,
        "thesis": thesis,
        "as_of_date": as_of_date,
    }
    final = await invoke_with_telemetry(graph, state)

    # 4. Score against realised prices
    score = score_run(
        ticker=ticker,
        as_of_date=as_of_date,
        horizons=horizons,
        state=final,
    )

    duration_s = time.perf_counter() - started_at

    # 5. Persist
    record = {
        "ticker": ticker,
        "thesis_slug": slug,
        "as_of_date": as_of_date,
        "horizons": horizons,
        "duration_s": round(duration_s, 1),
        "run_id": final.get("run_id"),
        "report": final.get("report") or "",
        "synthesis_confidence": final.get("synthesis_confidence"),
        "synthesis_verdict": final.get("synthesis_verdict"),
        "risk": final.get("risk") or {},
        "monte_carlo": final.get("monte_carlo") or {},
        "score": score,
        "errors": final.get("errors") or [],
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{ticker}__{as_of_date}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))
    logger.info(
        f"[backtest] {ticker} as_of={as_of_date}: "
        f"verdict={score.get('verdict')} conf={record['synthesis_confidence']} "
        f"duration={duration_s:.1f}s → {out_path}"
    )
    return record


async def _ensure_filings_ingested(ticker: str) -> None:
    """If `ticker` has zero chunks in the filings collection, download +
    ingest first. Production-mode ingestion (no as_of) — the per-call
    `chroma.query(..., as_of=...)` filter does the date filtering at
    retrieval time, which is the right place.

    Backtest mode in `data.edgar.download_filings` refuses to fetch from
    EDGAR when as_of is set (avoids leakage), so we MUST ingest without
    as_of. The metadata `filed_date` per chunk is what the as_of filter
    uses.
    """
    from data.chroma import _get_collection, ingest_filing
    from data.edgar import DEFAULT_LIMITS, download_filings

    coll = _get_collection()
    probe = coll.get(where={"ticker": ticker.upper()}, limit=1, include=[])
    if probe.get("ids"):
        logger.info(f"[backtest] {ticker}: filings already in chroma — skipping ingest")
        return

    logger.info(f"[backtest] {ticker}: filings corpus empty — ingesting now (no as_of)")
    # Bumped limits so we have ≥3 years of history; the as_of filter narrows.
    bumped_limits = {"10-K": 4, "10-Q": 12}
    paths = await download_filings(ticker, limits=bumped_limits)
    for p in paths:
        try:
            await asyncio.to_thread(ingest_filing, ticker, p)
        except Exception as e:
            logger.error(f"[backtest] {ticker} {p.name}: ingest failed: {e}")
    logger.info(f"[backtest] {ticker}: ingested {len(paths)} filings")
