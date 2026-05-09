"""Filings agent — LLM-driven thesis-aware analysis of SEC filings via hybrid RAG.

Pipeline per drill-in:
  1. Build 3 thesis-aware subqueries (Risk Factors / MD&A / Segment performance).
  2. Run hybrid retrieval (semantic + BM25 + RRF) for each, top-8 chunks each.
  3. One LLM call synthesises all 24 chunks into FilingsOutput.

The exact model is configured by the MODEL_FILINGS env var (see utils/models.py).

Run standalone:  python -m agents.filings NVDA [thesis_slug]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

from data.chroma import query as chroma_query
from utils import logger
from utils.models import MODEL_FILINGS
from utils.openrouter import get_client
from utils.schemas import FilingsOutput
from utils.state import FinaqState

NODE = "filings"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "filings.md").read_text()
LLM_MAX_TOKENS = 3000
SUBQUERY_K = 8
CANDIDATE_POOL = 60


# --- Subquery templates ------------------------------------------------------


def _build_subqueries(ticker: str, thesis: dict) -> list[dict]:
    """Three hardcoded thesis-aware subqueries.

    Each declares: label, item_filter (or None for whole-filing), question.
    """
    thesis_name = thesis.get("name", "the active thesis")
    return [
        {
            "label": "risk_factors",
            "item_filter": "1A",
            "question": (
                f"Principal risks to {ticker} viewed through the {thesis_name} thesis. "
                "Focus on supply, competition, customer concentration, geopolitical, "
                "and structural risks that affect the thesis."
            ),
        },
        {
            "label": "mdna_trajectory",
            "item_filter": "7",
            "question": (
                f"Recent trajectory of {ticker}'s revenue, margins, and segment "
                f"performance under the {thesis_name} thesis. Note any forward-looking "
                "guidance, capacity constraints, or pricing-power signals."
            ),
        },
        {
            "label": "segment_performance",
            "item_filter": None,
            "question": (
                f"How are {ticker}'s business segments performing? Identify the segments "
                f"most relevant to the {thesis_name} thesis and any disclosed segment-level "
                "capex, demand, or supply commentary."
            ),
        },
    ]


def _retrieve_for_subquery(
    ticker: str,
    subquery: dict,
    *,
    as_of: str | None = None,
) -> list[dict]:
    return chroma_query(
        ticker,
        subquery["question"],
        k=SUBQUERY_K,
        item_filter=subquery["item_filter"],
        candidate_pool=CANDIDATE_POOL,
        as_of=as_of,
    )


# --- Prompt assembly ---------------------------------------------------------


def _format_chunk(idx: int, chunk: dict) -> str:
    md = chunk.get("metadata") or {}
    filed_date = md.get("filed_date") or "unknown"
    accession = md.get("accession", "")
    item_label = md.get("item_label", "")
    text = chunk.get("text", "")
    return (
        f"\n[chunk {idx}] accession={accession} item={item_label!r} "
        f"filed_date={filed_date}\n{text}"
    )


def _build_user_prompt(
    ticker: str,
    thesis: dict,
    subqueries_with_chunks: list[tuple[dict, list[dict]]],
    *,
    as_of_date: str | None = None,
) -> str:
    today_iso = as_of_date or date.today().isoformat()
    parts = [
        f"AS OF: {today_iso}",
        f"TICKER: {ticker}",
        f"ACTIVE THESIS: {thesis.get('name', 'unknown')}",
        f"THESIS SUMMARY: {thesis.get('summary', '')}",
        f"ANCHOR TICKERS: {', '.join(thesis.get('anchor_tickers', []))}",
        "",
        "RETRIEVED FILING CHUNKS BY SUBQUERY:",
    ]
    for sq, chunks in subqueries_with_chunks:
        parts.append(f"\n=== SUBQUERY: {sq['label']} ===")
        parts.append(f"QUESTION: {sq['question']}")
        if not chunks:
            parts.append("(no chunks retrieved)")
            continue
        for i, ch in enumerate(chunks, start=1):
            parts.append(_format_chunk(i, ch))
    parts.append("\nPRODUCE YOUR ANALYSIS NOW. STRICT JSON ONLY.")
    return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text


def _call_llm(
    ticker: str,
    thesis: dict,
    subqueries_with_chunks: list[tuple[dict, list[dict]]],
    *,
    as_of_date: str | None = None,
) -> dict:
    from utils.as_of import maybe_inject_as_of

    client = get_client()
    user = _build_user_prompt(ticker, thesis, subqueries_with_chunks, as_of_date=as_of_date)
    system = maybe_inject_as_of(SYSTEM_PROMPT, as_of_date)
    resp = client.chat.completions.create(
        model=MODEL_FILINGS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(_strip_code_fences(raw))


# --- Graph node --------------------------------------------------------------


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    ticker = state.get("ticker", "")
    thesis = state.get("thesis") or {}
    as_of_date = state.get("as_of_date")  # backtest mode if non-None
    errors: list[str] = []

    # Step 1 — 3 RAG subqueries (each off-loop because chroma + BM25 are sync)
    subqueries = _build_subqueries(ticker, thesis)
    subqueries_with_chunks: list[tuple[dict, list[dict]]] = []
    for sq in subqueries:
        try:
            chunks = await asyncio.to_thread(
                _retrieve_for_subquery, ticker, sq, as_of=as_of_date,
            )
        except Exception as e:
            logger.error(f"[filings] retrieval failed for {sq['label']}: {e}")
            errors.append(f"retrieval/{sq['label']}: {e}")
            chunks = []
        subqueries_with_chunks.append((sq, chunks))

    total_chunks = sum(len(c) for _, c in subqueries_with_chunks)
    if total_chunks == 0:
        # Three failure modes that look the same on the wire:
        #   (a) Foreign issuer — files 20-F/6-K, not 10-K/10-Q. Our pipeline
        #       only ingests the latter two, so the corpus is empty AND the
        #       fix is NOT to re-run scripts.ingest_universe (won't help).
        #       Detected when the EDGAR cache shows other-kind directories.
        #   (b) Ticker not yet ingested → empty corpus, actionable: run script.
        #   (c) Ticker IS ingested but the 3 subqueries all returned 0
        #       (rare; would mean the subquery is poorly matched to content).
        # The Mission Control "Recent errors" panel + dashboard banner key
        # off these messages, so they need to be precise.
        from data.chroma import has_ticker
        from data.edgar import has_filings_in_unsupported_kinds

        ticker_ingested = await asyncio.to_thread(has_ticker, ticker)
        unsupported_kinds = await asyncio.to_thread(
            has_filings_in_unsupported_kinds, ticker
        )

        if not ticker_ingested and unsupported_kinds:
            # (a) Foreign issuer / wrong filing kinds present
            kinds_str = ", ".join(unsupported_kinds)
            specific_error = (
                f"foreign_issuer: {ticker} files {kinds_str} (not 10-K/10-Q). "
                f"Our pipeline only ingests 10-K + 10-Q today — see "
                f"docs/POSTPONED.md for 20-F/6-K support roadmap. "
                f"Re-running scripts.ingest_universe will NOT help."
            )
            summary = (
                f"{ticker} is a foreign-issuer or non-standard-kind filer "
                f"({kinds_str}). The current pipeline ingests 10-K + 10-Q only, "
                f"so {ticker}'s SEC filings are NOT in ChromaDB. This is a "
                f"known limitation tracked in `docs/POSTPONED.md`."
            )
        elif not ticker_ingested:
            # (b) Ticker just hasn't been ingested yet
            specific_error = (
                f"ticker_not_ingested: {ticker} has zero chunks in ChromaDB. "
                f"Run `python -m scripts.ingest_universe {ticker}` "
                f"(or `--thesis <slug>` for the whole thesis universe)."
            )
            summary = (
                f"No filings data for {ticker}. ChromaDB has not been ingested "
                f"for this ticker — run `python -m scripts.ingest_universe "
                f"{ticker}` to populate, then re-run the drill-in."
            )
        else:
            # (c) Ingested, but subqueries all missed
            specific_error = (
                f"empty_query_match: {ticker} is ingested but the subqueries "
                "matched no chunks. Check the subquery templates against "
                "the actual filing content."
            )
            summary = (
                f"{ticker} is ingested but the 3 thesis-aware subqueries "
                "returned no relevant chunks. The retrieval pool may be too "
                "narrow for this thesis × ticker pair."
            )
        out = FilingsOutput(
            summary=summary,
            risk_themes=[],
            mdna_quotes=[],
            evidence=[],
            errors=errors + [specific_error],
        )
    else:
        # Step 2 — LLM synthesis (off-loop)
        try:
            llm_out = await asyncio.to_thread(
                _call_llm, ticker, thesis, subqueries_with_chunks,
                as_of_date=as_of_date,
            )
            out = FilingsOutput.model_validate(llm_out)
            out.errors = errors
        except Exception as e:
            logger.error(f"[filings] LLM synthesis failed for {ticker}: {e}")
            errors.append(f"llm: {e}")
            out = FilingsOutput(
                summary=f"LLM synthesis failed for {ticker}; retrieval succeeded.",
                risk_themes=[],
                mdna_quotes=[],
                evidence=[],
                errors=errors,
            )

    # Stash the retrieval audit on the agent's payload so a post-completion
    # sidecar (utils/live_eval.evaluate_filings_retrieval) can grade
    # retrieval quality without re-running RAG. Only kept when there's
    # something to grade — keeps the cached state JSON lean otherwise.
    payload = out.model_dump()
    if total_chunks > 0:
        payload["_retrieval_audit"] = [
            {
                "label": sq.get("label"),
                "question": sq.get("question"),
                "chunks": [
                    {"text": c.get("text", ""), "metadata": c.get("metadata") or {}}
                    for c in chunks
                ],
            }
            for sq, chunks in subqueries_with_chunks
        ]

    return {
        "filings": payload,
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }


# --- CLI ---------------------------------------------------------------------


async def _cli(ticker: str, thesis_slug: str = "ai_cake") -> None:
    thesis = json.loads(Path(f"theses/{thesis_slug}.json").read_text())
    state: FinaqState = {"ticker": ticker, "thesis": thesis}
    result = await run(state)
    print(json.dumps(result["filings"], indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agents.filings TICKER [thesis_slug]", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()
    thesis_slug = sys.argv[2] if len(sys.argv) > 2 else "ai_cake"
    asyncio.run(_cli(ticker, thesis_slug))
