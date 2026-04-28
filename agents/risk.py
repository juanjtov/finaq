"""Risk agent — Step 4 stub.

Real implementation (synthesis-only, no external calls) lands in Step 5d.
This node sits at the fan-in barrier — it runs once fundamentals, filings,
and news have all completed.
"""

from __future__ import annotations

import time

from utils.state import FinaqState

NODE = "risk"


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    return {
        "risk": {
            "score_0_to_10": 5,
            "top_risks": [
                {"title": "[stub] macro risk", "severity": 3, "explanation": "stub"},
            ],
            "summary": f"[stub] consolidated risk view for {state.get('ticker', '?')}",
        },
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }
