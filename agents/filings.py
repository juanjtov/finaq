"""Filings agent — Step 4 stub. Real RAG implementation lands in Step 5b."""

from __future__ import annotations

import asyncio
import time

from utils.state import FinaqState

NODE = "filings"


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    await asyncio.sleep(0.05)
    ticker = state.get("ticker", "?")
    return {
        "filings": {
            "summary": f"[stub] filings synthesis for {ticker}",
            "risk_themes": ["macro", "supply chain", "concentration"],
            "mdna_quotes": [],
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
