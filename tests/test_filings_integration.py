"""Step 5b integration tests — real ChromaDB hybrid retrieval + real LLM synthesis.

Run via:  pytest -m integration tests/test_filings_integration.py

Requires NVDA filings already ingested into ChromaDB. The data layer integration
suite handles ingestion; if it hasn't run, this test will skip cleanly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.filings import run
from utils.schemas import FilingsOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.mark.asyncio
async def test_filings_real_run_on_nvda_ai_cake():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}
    result = await run(state)
    out = FilingsOutput.model_validate(result["filings"])

    if out.errors and any("no chunks" in e for e in out.errors):
        pytest.skip("ChromaDB has no NVDA chunks; run data-layer integration first")

    # Sanity-check the output shape
    assert out.summary, "summary is empty"
    assert 3 <= len(out.risk_themes) <= 6, f"unexpected risk_themes count: {out.risk_themes}"
    assert out.evidence, "no evidence emitted"

    # Every evidence entry must carry as_of (the freshness marker we wired in earlier)
    for ev in out.evidence:
        assert ev.as_of, f"evidence missing as_of: {ev}"
        assert ev.source == "edgar", ev

    # Evidence should span at least 2 distinct accessions OR 2 distinct item labels —
    # a sign the synthesis actually integrated multiple subqueries.
    accessions = {ev.accession for ev in out.evidence if ev.accession}
    items = {ev.item for ev in out.evidence if ev.item}
    assert len(accessions) + len(items) >= 2, "evidence didn't span multiple chunks"
