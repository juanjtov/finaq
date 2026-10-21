"""Thesis dicts shared by the unit tests.

`theses/*.json` is user-managed — theses get added, renamed, promoted and
archived over time — so no unit test reads one. These dicts mirror the three
Phase-0 hand-written theses (now under `theses/archive/`) that the agent
prompts and the Telegram flows were built against, trimmed to the parts the
tests assert on: `ai_cake` carries anchors, relationships, thresholds and a
valuation block; `nvda_halo` and `construction` share tickers with it (the
ambiguous-ticker flow); `general` is the universe-less catch-all.

Use them through the `ai_cake` / `theses_dir` fixtures in conftest.py, or
deep-copy a dict where a module-level helper needs one.
"""

from __future__ import annotations

import json
from pathlib import Path

AI_CAKE: dict = {
    "name": "AI cake",
    "summary": (
        "Picks-and-shovels across AI silicon, hyperscalers, networking and power; "
        "the power and cooling layer is structurally undersupplied."
    ),
    "anchor_tickers": ["NVDA", "MSFT"],
    "universe": ["NVDA", "AVGO", "TSM", "MSFT", "GOOGL", "ANET", "VRT", "CEG", "PWR"],
    "relationships": [
        {"from": "NVDA", "to": "TSM", "type": "supplier", "note": "TSM fabs NVDA's GPUs"},
        {"from": "NVDA", "to": "VRT", "type": "customer", "note": "VRT cools NVDA-spec racks"},
        {
            "from": "MSFT",
            "to": "CEG",
            "type": "customer",
            "note": "MSFT signed nuclear PPAs with CEG",
        },
    ],
    "material_thresholds": [
        {"signal": "data_center_capex_announcement", "operator": ">", "value": 5e9, "unit": "USD"},
        {"signal": "gross_margin_change_qoq", "operator": "abs >", "value": 200, "unit": "bps"},
        {
            "signal": "filing_mentions",
            "operator": "contains",
            "value": "capacity constraint",
            "unit": "text",
        },
        {"signal": "fcf_yield", "operator": "<", "value": 4, "unit": "percent"},
        {"signal": "capex_to_revenue_5yr_avg", "operator": ">", "value": 25, "unit": "percent"},
    ],
    "valuation": {
        "equity_risk_premium": 0.05,
        "erp_basis": "Long-run S&P 500 ERP plus a premium for AI capex cyclicality",
        "terminal_growth_rate": 0.03,
        "terminal_growth_basis": "Semiconductor and hyperscaler infrastructure long-run growth",
        "discount_rate_floor": 0.075,
        "discount_rate_cap": 0.13,
    },
}

NVDA_HALO: dict = {
    "name": "Halo · NVDA",
    "summary": (
        "Names whose fortunes track NVDA's data-center guide: server builders, "
        "power and cooling, networking silicon."
    ),
    "anchor_tickers": ["SMCI", "DELL"],
    "universe": ["SMCI", "DELL", "VRT", "CEG", "ANET", "MRVL"],
    "relationships": [
        {"from": "SMCI", "to": "DELL", "type": "peer", "note": "Both build NVIDIA-spec AI servers"},
        {
            "from": "ANET",
            "to": "MRVL",
            "type": "supplier",
            "note": "MRVL electro-optics in ANET switches",
        },
    ],
    "material_thresholds": [
        {
            "signal": "nvda_data_center_revenue_qoq",
            "operator": "abs >",
            "value": 5,
            "unit": "percent",
        },
        {
            "signal": "filing_mentions",
            "operator": "contains",
            "value": "supply constrained",
            "unit": "text",
        },
        {"signal": "fcf_yield", "operator": "<", "value": 4, "unit": "percent"},
    ],
    "valuation": {
        "equity_risk_premium": 0.06,
        "erp_basis": "S&P long-run ERP plus a premium for single-anchor concentration",
        "terminal_growth_rate": 0.025,
        "terminal_growth_basis": "Reverts to broad IT-sector long-run growth",
        "discount_rate_floor": 0.08,
        "discount_rate_cap": 0.14,
    },
}

CONSTRUCTION: dict = {
    "name": "Construction",
    "summary": (
        "Electrical contractors, equipment and grid builders riding the data-center "
        "and reshoring build-out."
    ),
    "anchor_tickers": ["CAT", "EME"],
    "universe": ["CAT", "EME", "PWR", "VRT", "URI", "MLM"],
    "relationships": [
        {
            "from": "EME",
            "to": "VRT",
            "type": "supplier",
            "note": "EME installs VRT cooling systems",
        },
        {
            "from": "CAT",
            "to": "URI",
            "type": "customer",
            "note": "URI rents out CAT equipment fleets",
        },
    ],
    "material_thresholds": [
        {"signal": "backlog_change_qoq", "operator": "abs >", "value": 10, "unit": "percent"},
        {
            "signal": "filing_mentions",
            "operator": "contains",
            "value": "labor shortage",
            "unit": "text",
        },
        {"signal": "fcf_yield", "operator": "<", "value": 4, "unit": "percent"},
    ],
    "valuation": {
        "equity_risk_premium": 0.055,
        "erp_basis": "S&P long-run ERP plus a premium for construction cyclicality",
        "terminal_growth_rate": 0.025,
        "terminal_growth_basis": "Long-run nominal construction spend growth",
        "discount_rate_floor": 0.075,
        "discount_rate_cap": 0.13,
    },
}

GENERAL: dict = {
    "name": "General equity research (Buffett framework)",
    "summary": (
        "Universe-less fallback for ad-hoc tickers: business quality, capital "
        "allocation, balance sheet and margin of safety."
    ),
    "anchor_tickers": [],
    "universe": [],
    "relationships": [],
    "material_thresholds": [
        {"signal": "roe_ttm", "operator": "<", "value": 12, "unit": "percent"},
        {"signal": "roic_ttm", "operator": "<", "value": 10, "unit": "percent"},
        {"signal": "debt_to_equity", "operator": ">", "value": 0.5, "unit": "ratio"},
        {"signal": "interest_coverage", "operator": "<", "value": 5, "unit": "x"},
        {
            "signal": "filing_mentions",
            "operator": "contains",
            "value": "going concern",
            "unit": "text",
        },
        {
            "signal": "filing_mentions",
            "operator": "contains",
            "value": "material weakness",
            "unit": "text",
        },
    ],
    "valuation": {
        "equity_risk_premium": 0.05,
        "erp_basis": "Long-run S&P 500 ERP, no thesis-specific tilt",
        "terminal_growth_rate": 0.025,
        "terminal_growth_basis": "US real long-run GDP trend",
        "discount_rate_floor": 0.07,
        "discount_rate_cap": 0.12,
    },
}

THESES: dict[str, dict] = {
    "ai_cake": AI_CAKE,
    "construction": CONSTRUCTION,
    "general": GENERAL,
    "nvda_halo": NVDA_HALO,
}


def write_theses(directory: Path) -> Path:
    """Write every fixture thesis to `directory/{slug}.json`; returns `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    for slug, data in THESES.items():
        (directory / f"{slug}.json").write_text(json.dumps(data, indent=2) + "\n")
    return directory
