"""Retrieval over the `synthesis_reports` corpus (Pinecone reports index).

The corpus is populated by `scripts.index_existing_reports` (backfill of
`data_cache/demos/*.json`). Each document is one section of a Synthesis
report (`## What this means`, `## Bull case`, ...) so the planner can
retrieve at the granularity of "the bull case from the prior NVDA / ai_cake
drill" without having to re-parse the full report.

Retrieval reuses `data.vectors`' tokeniser + BM25 + RRF helpers on top of
the reports index; the ticker-scoped lookups (`latest_watchlist_section`,
`latest_report_excerpts`) list a ticker's chunks by id prefix and need no
embedding call.

Output shape:
  list[{
    "text":     str,       # the section markdown
    "metadata": dict,      # {run_id, ticker, thesis, section, date, ...}
    "score":    float|None,
  }]
"""

from __future__ import annotations

from data.vectors import _bm25_rank, _reciprocal_rank_fusion, fetch_reports, query_reports

DEFAULT_K = 5
DEFAULT_CANDIDATE_POOL = 25


def _build_where(ticker: str | None, thesis: str | None) -> dict | None:
    """Compose the metadata pre-filter for the synthesis_reports schema.

    Schema: `{run_id, ticker, thesis, section, date, confidence, risk_level, ...}`.
    The planner usually pre-filters by `ticker` (always) and `thesis`
    (when present) — no item_code here, that was filings-specific.
    """
    conds: list[dict] = []
    if ticker:
        conds.append({"ticker": ticker.upper()})
    if thesis:
        conds.append({"thesis": thesis})
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


def query_past_reports(
    question: str,
    *,
    ticker: str | None = None,
    thesis: str | None = None,
    k: int = DEFAULT_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    use_keyword: bool = True,
) -> list[dict]:
    """Retrieve top-`k` section chunks from prior drill-ins relevant to `question`.

    Hybrid retrieval: metadata pre-filter → semantic top-N → BM25 over the
    same pool → Reciprocal Rank Fusion → top-`k`. Same recipe as the
    filings RAG pipeline, just on a different index.

    Returns `[]` (not None) on empty or missing corpus — the planner
    treats no past reports the same as "drill from scratch".
    """
    candidates = query_reports(question, where=_build_where(ticker, thesis), top_k=candidate_pool)
    if not candidates:
        return []

    semantic_indices = list(range(len(candidates)))
    if use_keyword and len(candidates) > 1:
        keyword_indices = _bm25_rank([c["text"] for c in candidates], question)
        fused = _reciprocal_rank_fusion([semantic_indices, keyword_indices])
    else:
        fused = semantic_indices

    return [candidates[i] for i in fused[:k]]


def _date_of(chunk: dict) -> str:
    m = chunk.get("metadata") or {}
    return str(m.get("date") or m.get("filed_at_iso") or "")


def latest_watchlist_section(
    *,
    ticker: str,
    thesis: str | None = None,
) -> dict | None:
    """Pull the most-recent `## Watchlist` chunk for the (ticker, thesis) pair.

    The Synthesis agent emits forward-looking signals to track in this
    section (e.g. "Q3 earnings call (Aug 2026) — listen for AI capex
    guidance (news)"). The CIO planner uses these as a hit-list: if
    recent news or filings match a watchlist item, that's a strong
    drill signal — the prior drill explicitly flagged it.

    Returns `{text, metadata}` for the latest watchlist chunk, or None
    when no past drill has produced one yet (or the index is empty).
    """
    try:
        chunks = fetch_reports(ticker, thesis=thesis, section="Watchlist", limit=50)
    except Exception:
        return None
    if not chunks:
        return None
    best = max(chunks, key=_date_of)
    return {"text": best["text"], "metadata": best["metadata"]}


def latest_report_excerpts(
    *,
    ticker: str,
    thesis: str | None = None,
    k: int = 3,
) -> list[dict]:
    """Quick "what does the most recent report say" view — pulls the top
    `k` sections of the most recent drill-in for a (ticker, thesis) pair.

    The planner uses this when the user invokes `/cio TICKER` and we
    want to give the LLM a single report's worth of context, not a
    cross-report RAG cocktail.

    Implementation: pull every chunk for the pair, group by `run_id`,
    pick the run_id with the most-recent `date`, return its first `k`
    sections (sorted by section weight: What this means → Thesis →
    Top risks first).
    """
    if not ticker:
        return []
    try:
        chunks = fetch_reports(ticker, thesis=thesis, limit=200)
    except Exception:
        return []

    # Group by run_id; pick most recent by date.
    by_run: dict[str, list[dict]] = {}
    for chunk in chunks:
        run_id = str((chunk.get("metadata") or {}).get("run_id") or "")
        if run_id:
            by_run.setdefault(run_id, []).append(chunk)
    if not by_run:
        return []

    most_recent_run = max(by_run, key=lambda rid: _date_of(by_run[rid][0]))
    sections = by_run[most_recent_run]

    # Section ordering — most decision-relevant first. Anything not in
    # the priority list gets pushed to the bottom (stable sort).
    priority = {
        "What this means": 0,
        "Thesis statement": 1,
        "Top risks": 2,
        "Action recommendation": 3,
        "Monte Carlo fair value": 4,
        "Bull case": 5,
        "Bear case": 6,
        "Watchlist": 7,
        "Evidence": 8,
    }
    sections.sort(key=lambda s: priority.get(str(s["metadata"].get("section") or ""), 99))
    return sections[:k]
