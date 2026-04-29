"""Tier 1c — output sanity tests with real Fundamentals + real treasury.

Run via:  pytest -m integration tests/test_monte_carlo_sanity.py

Catches catastrophic input bugs that all-stub tests would miss: e.g. a
Fundamentals refactor that renames `revenue_latest` would break MC silently;
this test fails loudly instead. Also asserts that `convergence_ratio` is
populated and reasonable on real-world data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents import monte_carlo
from agents.fundamentals import run as fundamentals_run

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.mark.asyncio
async def test_p50_within_sane_range_of_current_price():
    """Real Fundamentals → real MC: P50 should be within 0.2× to 5× current price.

    Catches gross misconfiguration (Fundamentals key rename, dilution disaster,
    discount rate < 0, etc.) that pure-math tests can't catch.
    """
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}

    fund_result = await fundamentals_run(state)
    state.update({"fundamentals": fund_result["fundamentals"]})

    mc_result = await monte_carlo(state)
    mc = mc_result["monte_carlo"]

    if mc.get("method") == "skipped":
        pytest.skip(f"MC node skipped: {mc.get('errors')}")

    current_price = mc["current_price"]
    dcf_p50 = mc["dcf"]["p50"]
    mult_p50 = mc["multiple"]["p50"]

    assert 0.2 * current_price < dcf_p50 < 5.0 * current_price, (
        f"DCF P50 ${dcf_p50:.2f} outside sane range vs current price ${current_price:.2f}; "
        "likely an input plumbing bug or Fundamentals projection blowing up"
    )
    assert (
        0.2 * current_price < mult_p50 < 5.0 * current_price
    ), f"Mult P50 ${mult_p50:.2f} outside sane range vs current price ${current_price:.2f}"


@pytest.mark.asyncio
async def test_convergence_ratio_emitted_and_reasonable_on_real_run():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis}

    fund_result = await fundamentals_run(state)
    state.update({"fundamentals": fund_result["fundamentals"]})

    mc_result = await monte_carlo(state)
    mc = mc_result["monte_carlo"]

    if mc.get("method") == "skipped":
        pytest.skip(f"MC node skipped: {mc.get('errors')}")

    cr = mc["convergence_ratio"]
    assert 0.0 <= cr <= 1.0, f"convergence_ratio out of [0,1]: {cr}"
    # On a typical large-cap, the two models shouldn't catastrophically diverge.
    # Convergence < 0.3 means one model is producing nonsense (e.g. negative
    # owner earnings every year of the DCF horizon).
    assert cr >= 0.3, (
        f"DCF and Multiple medians diverge severely (ratio={cr:.2f}); "
        "one model is mis-specified for this ticker"
    )


@pytest.mark.asyncio
async def test_discount_rate_used_within_thesis_band_on_real_run():
    thesis_dict = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {"ticker": "NVDA", "thesis": thesis_dict}

    fund_result = await fundamentals_run(state)
    state.update({"fundamentals": fund_result["fundamentals"]})

    mc_result = await monte_carlo(state)
    mc = mc_result["monte_carlo"]
    if mc.get("method") == "skipped":
        pytest.skip(f"MC node skipped: {mc.get('errors')}")

    val = thesis_dict["valuation"]
    assert val["discount_rate_floor"] <= mc["discount_rate_used"] <= val["discount_rate_cap"]
