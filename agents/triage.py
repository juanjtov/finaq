"""Triage agent — Phase 0 stub. Real implementation in Step 11.

Phase 0 returns an empty dict so the rest of the system has something to call;
the Streamlit "Run scan" button is fixture-backed until Step 11.
"""

from __future__ import annotations

from utils.state import FinaqState


async def run(state: FinaqState) -> dict:
    return {}
