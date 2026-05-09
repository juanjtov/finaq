"""LangGraph state definition for the FINAQ drill-in graph.

Each worker agent fills its corresponding key (fundamentals / filings / news /
risk). `messages` and `errors` use the operator.add reducer so multiple
parallel nodes can append without clobbering each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class FinaqState(TypedDict, total=False):
    ticker: str
    thesis: dict[str, Any]
    fundamentals: dict[str, Any]
    filings: dict[str, Any]
    news: dict[str, Any]
    risk: dict[str, Any]
    monte_carlo: dict[str, Any]
    report: str
    synthesis_confidence: str  # low|medium|high — duplicated from inside the markdown
    gaps: list[str]  # upstream content Synthesis wished it had (retrospective observability)
    watchlist: list[str]  # forward-looking events to track before next drill-in (per-agent suffix)
    run_id: str  # set by invoke_with_telemetry; lets the dashboard key cached files by run
    # Backtest mode (Step B1). When set to an ISO date string like "2025-09-05",
    # every agent treats data dated AFTER as_of_date as unavailable — yfinance
    # bars stop at as_of, EDGAR filings are filtered by filed_date, ChromaDB
    # chunks are filtered by metadata, news date-range queries cap at as_of,
    # and each agent's prompt gets an as-of context block. None (the default)
    # = production mode, behaviour is unchanged.
    as_of_date: str | None
    messages: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
