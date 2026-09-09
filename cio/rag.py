"""Retrieval over the `synthesis_reports` corpus (Pinecone reports index).

The corpus is populated by `scripts.index_existing_reports` (backfill of
`data_cache/demos/*.json`). Each document is one section of a Synthesis
report (`## What this means`, `## Bull case`, ...) so the planner can
retrieve at the granularity of "the bull case from the prior NVDA / ai_cake
drill" without having to re-parse the full report.

Retrieval reuses `data.vectors`' tokeniser + BM25 + RRF helpers on top of
the reports index; the ticker-scoped lookup (`latest_watchlist_section`)
lists a ticker's chunks by id prefix and needs no embedding call.

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
