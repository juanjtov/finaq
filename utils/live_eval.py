"""Per-drill-in RAG eval — opt-in sidecar that grades live Filings retrievals.

Different from `pytest -m eval` (which grades fixture state). This module
runs AFTER a real drill-in completes and fires Tier 2 LLM-judge calls
over the chunks the Filings agent actually retrieved for THIS user's
ticker × thesis. Results land in `data_cache/eval/runs/` keyed by the
drill-in's run_id, so Mission Control's eval surface includes
production-quality data alongside test-fixture data.

Gated by `EVAL_LIVE_DRILL_INS=true` in `.env` (default OFF). Cost is
~$0.012-0.025 per drill-in — 3 subqueries × ~8 chunks/subquery × ~$0.0005
per cheap-tier judge call (model resolved via `MODEL_JUDGE`). Fired in
a daemon thread by `ui/_runner.py`
after the drill-in's user-visible render completes, so the dashboard isn't
blocked.

The producer (Filings agent) records the retrieved chunks per subquery
on `state.filings._retrieval_audit` so this module can grade them
post-hoc without re-running RAG.
"""

from __future__ import annotations

import os
from typing import Any

from utils import logger
from utils.rag_eval import judge_relevance, serialise_judge_report, write_eval_run


def is_enabled() -> bool:
    """Read the env var that gates per-drill-in RAG eval. Default OFF —
    a personal-tool user shouldn't pay $0.02/drill-in unless they opt in."""
    return os.environ.get("EVAL_LIVE_DRILL_INS", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def evaluate_filings_retrieval(
    run_id: str | None,
    ticker: str,
    thesis_name: str,
    subqueries_with_chunks: list[tuple[dict, list[dict]]],
) -> list[dict]:
    """Fire Tier 2 LLM-judge over each subquery's retrieved chunks. Persist
    one eval run per subquery, keyed by `run_id` so the result links back
    to the originating drill-in.

    Returns the list of eval-run dicts that were persisted (for testing
    + dashboard surfacing).
    """
    persisted: list[dict] = []
    if not subqueries_with_chunks:
        return persisted

    for subquery, chunks in subqueries_with_chunks:
        if not chunks:
            # No chunks — nothing for the judge to grade. Persist an empty
            # row so Mission Control reflects "this subquery returned 0".
            row = {
                "tier": 2,
                "suite": "filings_live",
                "subuite": subquery.get("label", "?"),
                "ticker": ticker,
                "thesis": thesis_name,
                "run_id": run_id,
                "subquery_label": subquery.get("label", "?"),
                "subquery_question": subquery.get("question", ""),
                "k": 0,
                "precision_at_k": None,
                "ndcg_at_k": None,
                "mrr": None,
                "scores": [],
            }
            try:
                write_eval_run(row)
                persisted.append(row)
            except Exception as e:
                logger.warning(
                    f"[live_eval] persist failed for empty subquery "
                    f"{subquery.get('label')}: {e}"
                )
            continue

        question = subquery.get("question", "")
        try:
            report = judge_relevance(question, chunks)
        except Exception as e:
            logger.warning(
                f"[live_eval] judge_relevance failed for subquery "
                f"{subquery.get('label')}: {e}"
            )
            continue

        row: dict[str, Any] = {
            "tier": 2,
            "suite": "filings_live",
            "subsuite": subquery.get("label", "?"),
            "ticker": ticker,
            "thesis": thesis_name,
            "run_id": run_id,
            "subquery_label": subquery.get("label", "?"),
            "subquery_question": question,
            **serialise_judge_report(report),
        }
        try:
            write_eval_run(row)
            persisted.append(row)
        except Exception as e:
            logger.warning(
                f"[live_eval] persist failed for subquery "
                f"{subquery.get('label')}: {e}"
            )

    return persisted
