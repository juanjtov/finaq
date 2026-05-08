"""Retrieval over the `synthesis_reports` ChromaDB collection.

The collection is populated by `scripts.index_existing_reports` (backfill
of `data_cache/demos/*.json`) and — once Step 11.9's CIO orchestration
lands — by the runner sidecar that ships every fresh drill-in's report
into the collection in the same shape.

Each document is one section of a Synthesis report (`## What this means`,
`## Bull case`, ...) so the planner can retrieve at the granularity of
"the bull case from the prior NVDA / ai_cake drill" without having to
re-parse the full report.

Retrieval reuses `data.chroma`'s tokeniser + BM25 + RRF helpers, just
parameterised with `collection_name="synthesis_reports"`.

Output shape:
  list[{
    "text":     str,       # the section markdown
    "metadata": dict,      # {run_id, ticker, thesis, section, date, ...}
    "score":    float|None,
  }]
"""

from __future__ import annotations

from data.chroma import _bm25_rank, _get_collection, _reciprocal_rank_fusion

REPORTS_COLLECTION = "synthesis_reports"
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


def _unpack(results: dict) -> list[dict]:
    chunks: list[dict] = []
    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[None] * len(ids)])[0]
    for i in range(len(ids)):
        chunks.append(
            {
                "text": docs[i],
                "metadata": metas[i],
                "score": dists[i],
            }
        )
    return chunks


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

    Hybrid retrieval: ChromaDB metadata pre-filter → semantic top-N →
    BM25 over the same pool → Reciprocal Rank Fusion → top-`k`. Same
    recipe as the filings RAG pipeline, just on a different collection.

    Returns `[]` (not None) on empty or missing collection — the planner
    treats no past reports the same as "drill from scratch".
    """
    coll = _get_collection(name=REPORTS_COLLECTION)
    where = _build_where(ticker, thesis)

    sem_results = coll.query(
        query_texts=[question],
        n_results=candidate_pool,
        where=where,
    )
    candidates = _unpack(sem_results)
    if not candidates:
        return []

    semantic_indices = list(range(len(candidates)))
    if use_keyword and len(candidates) > 1:
        keyword_indices = _bm25_rank([c["text"] for c in candidates], question)
        fused = _reciprocal_rank_fusion([semantic_indices, keyword_indices])
    else:
        fused = semantic_indices

    return [candidates[i] for i in fused[:k]]


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
    when no past drill has produced one yet (or ChromaDB is empty).
    """
    coll = _get_collection(name=REPORTS_COLLECTION)
    conds: list[dict] = [{"ticker": ticker.upper()}]
    if thesis:
        conds.append({"thesis": thesis})
    conds.append({"section": "Watchlist"})
    
    #Boilerplate to keep where format the same {"$and":conds}
    where = conds[0] if len(conds) == 1 else {"$and": conds}

    try:
        results = coll.get(where=where, limit=50)
    except Exception:
        return None
    ids = results.get("ids") or []
    if not ids:
        return None
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    # Most-recent by metadata.date (filed_at_iso falls back).
    def _date_key(idx: int) -> str:
        m = metas[idx] or {}
        return str(m.get("date") or m.get("filed_at_iso") or "")

    best = max(range(len(ids)), key=_date_key)
    return {
        "text": docs[best],
        "metadata": metas[best],
    }


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
    coll = _get_collection(name=REPORTS_COLLECTION)
    where = _build_where(ticker, thesis)
    if not where:
        return []
    try:
        results = coll.get(where=where, limit=200)
    except Exception:
        return []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    if not ids:
        return []

    # Group by run_id; pick most recent by date.
    by_run: dict[str, list[dict]] = {}
    for i, _id in enumerate(ids):
        meta = metas[i] or {}
        run_id = str(meta.get("run_id") or "")
        if not run_id:
            continue
        by_run.setdefault(run_id, []).append(
            {"text": docs[i], "metadata": meta, "score": None}
        )

    if not by_run:
        return []
    # Sort run_ids by metadata.date descending (fallback: filed_at_iso).
    def _run_date(run_id: str) -> str:
        first = by_run[run_id][0]["metadata"]
        return str(first.get("date") or first.get("filed_at_iso") or "")

    most_recent_run = sorted(by_run.keys(), key=_run_date, reverse=True)[0]
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
    sections.sort(
        key=lambda s: priority.get(str(s["metadata"].get("section") or ""), 99)
    )
    return sections[:k]
