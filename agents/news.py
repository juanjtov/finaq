"""News agent — Step 4 stub. Real Tavily implementation lands in Step 5c."""

from __future__ import annotations

import asyncio
import time

from utils.state import FinaqState

NODE = "news"


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    await asyncio.sleep(0.05)
    ticker = state.get("ticker", "?")
    return {
        "news": {
            "summary": f"[stub] news scan for {ticker} (last 90d)",
            "catalysts": [],
            "concerns": [],
            "evidence": [],
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
