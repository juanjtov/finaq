"""Synthesis agent — Step 4 stub.

Real Opus implementation lands in Step 7. This stub emits a markdown skeleton
that mirrors the §11 section structure so downstream code (PDF export, Streamlit)
has something to render against.
"""

from __future__ import annotations

import time
from datetime import date

from utils.state import FinaqState

NODE = "synthesis"

REPORT_TEMPLATE = """# {ticker} — {thesis_name} thesis update

**Date:** {today} · **Confidence:** medium

## Thesis statement
[stub] thesis statement for {ticker}.

## Bull case
- [stub] bull point 1 (citation)
- [stub] bull point 2 (citation)
- [stub] bull point 3 (citation)

## Bear case
- [stub] bear point 1 (citation)
- [stub] bear point 2 (citation)
- [stub] bear point 3 (citation)

## Top risks
1. [stub] risk one — severity 3
2. [stub] risk two — severity 2

## Monte Carlo fair value
[stub] P10 / P50 / P90 placeholder.

## Action recommendation
[stub] no action recommended at this time.

## Evidence
- [stub] evidence list
"""


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    thesis_name = (state.get("thesis") or {}).get("name", "unknown")
    report = REPORT_TEMPLATE.format(
        ticker=state.get("ticker", "?"),
        thesis_name=thesis_name,
        today=date.today().isoformat(),
    )
    return {
        "report": report,
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }
