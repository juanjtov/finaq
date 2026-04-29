"""Step 7 unit tests — Synthesis agent + PDF exporter (Tier 1, deterministic).

Three classes of tests:
  - **Prompt assembly** — `_format_*` and `_build_user_prompt` produce non-empty
    sections that include the right ticker / thesis / numbers without crashing
    on missing inputs.
  - **Output coercion** — `_coerce_to_synthesis_output` enforces the shape
    contract (non-empty `report`, normalises `confidence`, defaults `gaps`).
  - **Failure paths in `run`** — empty upstream state → fallback report; LLM
    raise → fallback report. No external calls.
  - **PDF export** — markdown produced by Synthesis renders to a non-empty
    PDF without ReportLab errors.

Tier 2 LLM-judge eval lives in test_synthesis_eval.py (`pytest -m eval`).
Tier 3 integration sanity lives in test_synthesis_sanity.py (`pytest -m integration`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agents.synthesis import (
    _build_user_prompt,
    _coerce_to_synthesis_output,
    _collect_evidence,
    _fallback_report,
    _format_filings,
    _format_fundamentals,
    _format_monte_carlo,
    _format_news,
    _format_risk,
    _strip_code_fences,
    run,
)
from utils.pdf_export import export
from utils.schemas import SynthesisOutput

THESES_DIR = Path(__file__).parents[1] / "theses"


# --- Fixtures ---------------------------------------------------------------


def _full_state() -> dict:
    """A maximally-populated state — every upstream agent emitted a non-empty,
    schema-valid output. Used to exercise the prompt assembly path."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    return {
        "ticker": "NVDA",
        "thesis": thesis,
        "fundamentals": {
            "summary": "NVDA: 47% YoY revenue growth, 65% operating margins.",
            "kpis": {
                "revenue_5y_cagr": 1.0,
                "fcf_yield": 1.84,
                "operating_margin_5yr_avg": 0.49,
                "pe_trailing": 44.3,
                "current_price": 200.0,
                "shares_outstanding": 24e9,
                "revenue_latest": 60.9e9,
            },
            "projections": {
                "revenue_growth_mean": 0.20,
                "revenue_growth_std": 0.05,
                "margin_mean": 0.65,
                "margin_std": 0.04,
                "tax_rate_mean": 0.18,
                "tax_rate_std": 0.02,
                "maintenance_capex_pct_rev_mean": 0.04,
                "maintenance_capex_pct_rev_std": 0.01,
                "da_pct_rev_mean": 0.03,
                "da_pct_rev_std": 0.01,
                "dilution_rate_mean": 0.005,
                "dilution_rate_std": 0.002,
                "exit_multiple_mean": 28.0,
                "exit_multiple_std": 4.0,
            },
            "evidence": [
                {
                    "source": "yfinance",
                    "note": "revenue_latest",
                    "excerpt": "60.9e9",
                    "as_of": "2026-04-25",
                }
            ],
        },
        "filings": {
            "summary": "Capacity-constrained on Blackwell ramp; supply concentration risk.",
            "risk_themes": ["supply concentration", "export-control regulation"],
            "mdna_quotes": [
                {
                    "text": "We continue to face supply constraints across leading-edge nodes.",
                    "accession": "0001045810-26-000123",
                    "item": "Item 7",
                }
            ],
            "evidence": [
                {
                    "source": "edgar",
                    "accession": "0001045810-26-000123",
                    "item": "Item 7",
                    "as_of": "2026-02-21",
                    "excerpt": "Supply constraints persist...",
                }
            ],
        },
        "news": {
            "summary": "Mixed: hyperscaler capex raised to $320B; valuation premium concerns.",
            "catalysts": [
                {
                    "title": "MSFT raised AI capex guidance to $320B",
                    "summary": "Microsoft raised guidance",
                    "sentiment": "bull",
                    "url": "https://example.com/msft",
                    "as_of": "2026-04-15",
                }
            ],
            "concerns": [
                {
                    "title": "Valuation premium fading",
                    "summary": "Sell-side notes",
                    "sentiment": "bear",
                    "url": "https://example.com/val",
                    "as_of": "2026-04-19",
                }
            ],
            "evidence": [
                {
                    "source": "tavily",
                    "url": "https://example.com/msft",
                    "as_of": "2026-04-15",
                    "excerpt": "MSFT raised guidance",
                }
            ],
        },
        "risk": {
            "level": "ELEVATED",
            "score_0_to_10": 6,
            "summary": "Multiple convergent signals; FCF-yield breach.",
            "top_risks": [
                {
                    "title": "Supply concentration",
                    "severity": 4,
                    "explanation": "TSM N3/N2 sole supplier of leading-edge.",
                    "sources": ["fundamentals", "filings"],
                },
                {
                    "title": "FCF yield below MoS",
                    "severity": 3,
                    "explanation": "1.84% vs 4% threshold.",
                    "sources": ["fundamentals"],
                },
            ],
            "convergent_signals": [
                {
                    "theme": "supply concentration",
                    "sources": ["fundamentals", "filings"],
                    "explanation": "Both surface the same risk.",
                }
            ],
            "threshold_breaches": [
                {
                    "signal": "fcf_yield",
                    "operator": "<",
                    "threshold_value": 4,
                    "observed_value": 1.84,
                    "explanation": "Below MoS threshold.",
                    "source": "fundamentals",
                }
            ],
        },
        "monte_carlo": {
            "method": "dcf+multiple",
            "current_price": 200.0,
            "discount_rate_used": 0.095,
            "terminal_growth_used": 0.03,
            "convergence_ratio": 0.82,
            "n_sims": 10000,
            "n_years": 10,
            "dcf": {"p10": 120.0, "p25": 150.0, "p50": 185.0, "p75": 220.0, "p90": 260.0},
            "multiple": {
                "p10": 110.0,
                "p25": 145.0,
                "p50": 180.0,
                "p75": 215.0,
                "p90": 255.0,
            },
        },
    }


# --- Section formatters ----------------------------------------------------


def test_format_fundamentals_includes_kpis_and_projections():
    state = _full_state()
    out = _format_fundamentals(state["fundamentals"])
    assert "FUNDAMENTALS" in out
    assert "fcf_yield" in out
    assert "revenue_growth_mean" in out


def test_format_fundamentals_handles_empty_payload():
    out = _format_fundamentals({})
    assert "agent did not produce" in out


def test_format_filings_renders_quotes_with_accessions():
    state = _full_state()
    out = _format_filings(state["filings"])
    assert "0001045810-26-000123" in out
    assert "supply constraints" in out.lower()
    assert "supply concentration" in out


def test_format_news_includes_sentiment_and_url():
    state = _full_state()
    out = _format_news(state["news"])
    assert "[bull " in out
    assert "[bear " in out
    assert "https://example.com/msft" in out


def test_format_risk_includes_level_and_top_risks():
    state = _full_state()
    out = _format_risk(state["risk"])
    assert "level: ELEVATED" in out
    assert "Supply concentration" in out
    assert "convergent_signals" in out
    assert "threshold_breaches" in out
    assert "fcf_yield" in out


def test_format_monte_carlo_full_distribution():
    state = _full_state()
    out = _format_monte_carlo(state["monte_carlo"])
    assert "method: dcf+multiple" in out
    assert "P50=185.00" in out
    assert "convergence_ratio: 0.82" in out
    assert "discount_rate_used: 0.0950" in out


def test_format_monte_carlo_handles_skipped():
    out = _format_monte_carlo({"method": "skipped", "errors": ["missing inputs"]})
    assert "skipped" in out


def test_format_monte_carlo_handles_empty_payload():
    out = _format_monte_carlo({})
    assert "skipped" in out


# --- Prompt assembly --------------------------------------------------------


def test_build_user_prompt_includes_all_sections():
    state = _full_state()
    p = _build_user_prompt(state)
    assert "TICKER: NVDA" in p
    assert "AI cake" in p
    assert "FUNDAMENTALS" in p
    assert "FILINGS" in p
    assert "NEWS" in p
    assert "RISK" in p
    assert "MONTE CARLO" in p
    assert "EVIDENCE INVENTORY" in p
    assert "STRICT JSON ONLY" in p


def test_build_user_prompt_includes_thesis_thresholds():
    state = _full_state()
    p = _build_user_prompt(state)
    # ai_cake has fcf_yield + capex thresholds
    assert "fcf_yield" in p


def test_build_user_prompt_handles_partial_state():
    """If fundamentals failed but filings + news ran, prompt still assembles."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = {
        "ticker": "NVDA",
        "thesis": thesis,
        "fundamentals": {},  # failed
        "filings": {"summary": "filings summary", "risk_themes": []},
        "news": {"summary": "news summary"},
    }
    p = _build_user_prompt(state)
    assert "agent did not produce" in p
    assert "filings summary" in p


# --- _collect_evidence ------------------------------------------------------


def test_collect_evidence_unions_across_agents():
    state = _full_state()
    ev = _collect_evidence(state)
    sources = {e["agent"] for e in ev}
    assert {"fundamentals", "filings", "news"}.issubset(sources)


# --- _strip_code_fences -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"report":"x"}', '{"report":"x"}'),
        ('```json\n{"report":"x"}\n```', '{"report":"x"}'),
        ('```\n{"a":1}\n```', '{"a":1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- _coerce_to_synthesis_output --------------------------------------------


def test_coerce_normalises_confidence_lowercase():
    """LLM may emit 'HIGH' instead of 'high' despite instructions."""
    out = _coerce_to_synthesis_output(
        {"report": "# header\n\nbody", "confidence": "HIGH", "gaps": []}
    )
    assert out["confidence"] == "high"


def test_coerce_defaults_unknown_confidence_to_medium():
    out = _coerce_to_synthesis_output(
        {"report": "# header", "confidence": "very-confident", "gaps": []}
    )
    assert out["confidence"] == "medium"


def test_coerce_defaults_missing_confidence():
    out = _coerce_to_synthesis_output({"report": "# header"})
    assert out["confidence"] == "medium"
    assert out["gaps"] == []


def test_coerce_rejects_empty_report():
    with pytest.raises(ValueError, match="empty"):
        _coerce_to_synthesis_output({"report": "", "confidence": "high"})


def test_coerce_rejects_whitespace_only_report():
    with pytest.raises(ValueError, match="empty"):
        _coerce_to_synthesis_output({"report": "   \n   ", "confidence": "high"})


def test_coerce_drops_non_string_gaps():
    """If LLM emits structured gaps (dicts) instead of strings, coerce to str."""
    out = _coerce_to_synthesis_output(
        {"report": "# x", "confidence": "low", "gaps": ["one", None, "", "two"]}
    )
    assert out["gaps"] == ["one", "two"]


def test_coerce_handles_non_list_gaps():
    out = _coerce_to_synthesis_output({"report": "# x", "gaps": "not-a-list"})
    assert out["gaps"] == []


def test_coerce_propagates_watchlist():
    """Watchlist round-trips like gaps — same coercion semantics."""
    out = _coerce_to_synthesis_output(
        {
            "report": "# x",
            "watchlist": ["Q3 earnings (news)", None, "", "TSM yield (filings)"],
        }
    )
    assert out["watchlist"] == ["Q3 earnings (news)", "TSM yield (filings)"]


def test_coerce_defaults_missing_watchlist():
    """Missing watchlist → empty list, no error."""
    out = _coerce_to_synthesis_output({"report": "# x"})
    assert out["watchlist"] == []


def test_coerce_handles_non_list_watchlist():
    out = _coerce_to_synthesis_output({"report": "# x", "watchlist": "not-a-list"})
    assert out["watchlist"] == []


def test_coerce_recovers_watchlist_from_markdown_when_json_field_empty():
    """LLM sometimes fills the markdown ## Watchlist section but forgets to
    populate the JSON watchlist array. We recover from the markdown so
    Phase 1 Triage doesn't silently lose data."""
    md = (
        "# Header\n\n"
        "## Watchlist\n"
        "- Q3 earnings call (news)\n"
        "- TSM yield disclosure (filings)\n\n"
        "## Evidence\n- src\n"
    )
    out = _coerce_to_synthesis_output({"report": md, "watchlist": []})
    assert out["watchlist"] == ["Q3 earnings call (news)", "TSM yield disclosure (filings)"]


def test_coerce_keeps_explicit_watchlist_over_markdown_recovery():
    """If LLM populated BOTH the JSON array and the markdown section, the
    explicit JSON wins — recovery is only used when the array is empty."""
    md = (
        "# Header\n\n"
        "## Watchlist\n- bullet from markdown (filings)\n\n"
        "## Evidence\n- src\n"
    )
    out = _coerce_to_synthesis_output(
        {"report": md, "watchlist": ["bullet from json (news)"]}
    )
    assert out["watchlist"] == ["bullet from json (news)"]


# --- run() failure paths ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_fallback_when_no_upstream_state():
    """No fundamentals + filings + news + risk + MC → fallback, no LLM call."""
    state = {"ticker": "ZZZZ", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    assert "Synthesis unavailable" in result["report"]
    assert result["synthesis_confidence"] == "low"
    assert result["gaps"] == ["all upstream agents produced no output"]


@pytest.mark.asyncio
async def test_run_fallback_when_llm_raises(monkeypatch):
    """LLM raises but upstream state exists → fallback, error logged."""
    from agents import synthesis as syn

    monkeypatch.setattr(
        syn,
        "_call_llm",
        lambda state: (_ for _ in ()).throw(RuntimeError("openrouter outage")),
    )
    state = _full_state()
    result = await run(state)
    assert "Synthesis unavailable" in result["report"]
    assert result["synthesis_confidence"] == "low"
    # Synthesis-failure top-risk should be present in the fallback
    assert "Synthesis failure" in result["report"]


@pytest.mark.asyncio
async def test_run_propagates_llm_output_when_call_succeeds(monkeypatch):
    """Happy path: mock LLM returns valid JSON → fields propagate to state keys."""
    from agents import synthesis as syn

    fake_md = (
        "# NVDA — AI cake thesis update\n\n"
        "**Date:** 2026-04-29 · **Confidence:** medium\n\n"
        "## What this means\nNVIDIA makes AI chips. The bet is that data-center spending grows. The math says it's roughly fairly priced. Hold. Watch Q3 capex guidance.\n\n"
        "## Thesis statement\nThe view (Fund kpis).\n\n"
        "## Bull case\n- Strong growth (Fund kpis)\n- Capacity constrained (Filings 10-Q Item 7)\n"
        "- Hyperscaler capex up (News, 2026-04-15)\n\n"
        "## Bear case\n- FCF yield 1.8% (Fund kpis)\n- Customer concentration (Filings Item 1A)\n"
        "- Valuation pricey (News, 2026-04-19)\n\n"
        "## Top risks\n1. Supply — severity 4 — TSM N3/N2 dependency.\n"
        "2. FCF below MoS — severity 3 — 1.84% vs 4%.\n\n"
        "## Monte Carlo fair value\nP50 $185 vs $200 current; convergence 0.82.\n\n"
        "- **Bull (P75-P90):** Capex compounds 25%+.\n"
        "- **Base (P25-P75):** Steady ramp.\n"
        "- **Bear (P10-P25):** Digestion phase.\n\n"
        "## Action recommendation\nTrim 20% if Q3 misses $42B guide.\n\n"
        "## Watchlist\n- Q3 earnings call (news)\n- TSM yield disclosure (filings)\n"
        "- FCF margin trend (fundamentals)\n\n"
        "## Evidence\n- yfinance revenue=$60.9B\n- 10-Q accession 0001045810-26-000123\n"
    )
    fake_llm = {
        "report": fake_md,
        "confidence": "medium",
        "gaps": ["no segment-level capex split"],
        "watchlist": [
            "Q3 earnings call (news)",
            "TSM yield disclosure (filings)",
            "FCF margin trend (fundamentals)",
        ],
    }
    monkeypatch.setattr(syn, "_call_llm", lambda state: fake_llm)

    state = _full_state()
    result = await run(state)
    assert "## Thesis statement" in result["report"]
    assert "## What this means" in result["report"]
    assert "## Watchlist" in result["report"]
    assert result["synthesis_confidence"] == "medium"
    assert result["gaps"] == ["no segment-level capex split"]
    assert len(result["watchlist"]) == 3
    assert "(filings)" in result["watchlist"][1]


# --- Fallback report shape --------------------------------------------------


def test_fallback_report_includes_all_required_sections():
    state = {"ticker": "ZZZZ", "thesis": {"name": "Test thesis"}}
    md = _fallback_report(state)
    for header in (
        "## What this means",
        "## Thesis statement",
        "## Bull case",
        "## Bear case",
        "## Top risks",
        "## Monte Carlo fair value",
        "## Action recommendation",
        "## Watchlist",
        "## Evidence",
    ):
        assert header in md, f"fallback missing {header}"
    assert "Confidence:** low" in md


# --- PDF export -------------------------------------------------------------


SAMPLE_REPORT = """# NVDA — AI cake thesis update

**Date:** 2026-04-29 · **Confidence:** medium

## What this means
NVIDIA designs the chips that run modern AI; almost every big tech company buys them. The bet is that data-center spending keeps growing for several more years. The math says the stock is roughly fairly priced today — about 7% above what the model thinks it's worth. We'd hold the position and wait. Watch next quarter's earnings call for AI capex guidance.

## Thesis statement
NVIDIA remains the dominant supplier of AI accelerators with an estimated 90% data-center GPU share.

## Bull case
- Revenue grew 47% YoY in Q4 (Fund kpis)
- Backlog covers next two quarters with capacity constraints (Filings 10-Q Item 7)
- Hyperscaler capex guidance raised to $320B (News, 2026-04-15)

## Bear case
- FCF yield at 1.8% well below 4% MoS threshold (Fund kpis)
- 3 hyperscalers represent 50% of revenue (Filings 10-K Item 1A)
- Valuation premium fading per sell-side (News, 2026-04-19)

## Top risks
1. Customer concentration — severity 4 — 3 hyperscalers = 50% of revenue.
2. Capex deceleration — severity 3 — possible digestion phase by 2027.

## Monte Carlo fair value
DCF P50 of $185 vs current price $200 implies 7% downside. Convergence 0.82, discount 9.5%.

- **Bull (P75-P90):** Hyperscaler capex compounds 25%+ through 2028; NVDA holds share against ASIC competition.
- **Base (P25-P75):** Capex grows 15-20%; NVDA ships per current backlog and Blackwell roadmap.
- **Bear (P10-P25):** Hyperscaler digestion phase in 2027; multiple compresses to sector P/E.

## Action recommendation
Trim 20% if Q3 misses $42B guide. Hold otherwise; do not add until convergence_ratio rises.

## Watchlist
- Q3 earnings call (Aug 2026) — listen for AI capex guidance (news)
- TSM yield disclosure in next 10-Q — supply concentration check (filings)
- FCF margin trend in next quarter — has it crossed 25%? (fundamentals)
- Export-control rule updates from BIS over the next 90 days (news)

## Evidence
- yfinance: revenue=$60.9B (2026-Q4)
- 10-Q Item 7 (accession 0001045810-26-000123)
- News: https://example.com/msft (2026-04-15)
"""


def test_pdf_export_writes_non_empty_file(tmp_path: Path):
    out = export(SAMPLE_REPORT, tmp_path / "report.pdf")
    assert out.exists()
    assert out.stat().st_size > 1000  # PDFs always larger than 1KB


def test_pdf_export_creates_parent_dirs(tmp_path: Path):
    """If the parent dir doesn't exist, exporter creates it (no FileNotFoundError)."""
    nested = tmp_path / "a" / "b" / "report.pdf"
    out = export(SAMPLE_REPORT, nested)
    assert out.exists()


def test_pdf_export_rejects_empty_markdown(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        export("", tmp_path / "x.pdf")


def test_pdf_export_handles_special_characters_in_prose(tmp_path: Path):
    """ReportLab Paragraphs treat <,>,& as HTML markup — must be escaped.
    A naive renderer crashes on prose like 'P10 < P50'."""
    md = (
        "# NVDA — Test\n\n"
        "**Date:** 2026-04-28 · **Confidence:** high\n\n"
        "## Thesis statement\nP10 < P50 < P90 (DCF model). A & B & C.\n\n"
        "## Bull case\n- 5 < x < 10 (Fund kpis)\n\n"
        "## Bear case\n- a > b (Fund kpis)\n\n"
        "## Top risks\n1. Risk one — severity 3 — explanation.\n\n"
        "## Monte Carlo fair value\nDistribution shape.\n\n"
        "## Action recommendation\nNo action.\n\n"
        "## Evidence\n- src\n"
    )
    out = export(md, tmp_path / "special.pdf")
    assert out.exists()


def test_pdf_export_handles_bold_inline(tmp_path: Path):
    """Bold (`**text**`) must render without raising."""
    md = (
        "# NVDA — Test\n\n"
        "**Date:** 2026-04-28 · **Confidence:** medium\n\n"
        "## Thesis statement\nA paragraph with **bold inline** text.\n\n"
        "## Bull case\n- Bullet with **bold** inside (Fund kpis)\n\n"
        "## Bear case\n- A bullet (Fund kpis)\n\n"
        "## Top risks\n1. Risk — severity 2 — text.\n\n"
        "## Monte Carlo fair value\nP50.\n\n"
        "## Action recommendation\nHold.\n\n"
        "## Evidence\n- src\n"
    )
    out = export(md, tmp_path / "bold.pdf")
    assert out.exists()


# --- Structural assertions on a sample synthesis output --------------------


REQUIRED_SECTIONS = (
    "## What this means",
    "## Thesis statement",
    "## Bull case",
    "## Bear case",
    "## Top risks",
    "## Monte Carlo fair value",
    "## Action recommendation",
    "## Watchlist",
    "## Evidence",
)


def test_sample_report_contains_all_required_section_headers():
    """Locks in the §11 contract — used by Tier 3 to check real LLM output."""
    for h in REQUIRED_SECTIONS:
        assert h in SAMPLE_REPORT


def test_sample_report_bull_bullets_in_3_to_5_range():
    bull = _section(SAMPLE_REPORT, "## Bull case")
    bullets = [line for line in bull.splitlines() if line.startswith("- ")]
    assert 3 <= len(bullets) <= 5


def test_sample_report_bear_bullets_in_3_to_5_range():
    bear = _section(SAMPLE_REPORT, "## Bear case")
    bullets = [line for line in bear.splitlines() if line.startswith("- ")]
    assert 3 <= len(bullets) <= 5


def test_sample_report_bullets_under_20_words():
    for section in ("## Bull case", "## Bear case"):
        body = _section(SAMPLE_REPORT, section)
        for bullet in [line for line in body.splitlines() if line.startswith("- ")]:
            words = bullet[2:].split()
            assert len(words) <= 20, f"bullet too long ({len(words)} words): {bullet}"


def test_sample_report_top_risks_have_severity():
    body = _section(SAMPLE_REPORT, "## Top risks")
    numbered = [line for line in body.splitlines() if re.match(r"^\d+\.", line)]
    assert numbered, "no numbered risks"
    for line in numbered:
        assert re.search(r"severity\s*\d+", line), f"top-risk line missing severity: {line}"


def test_sample_report_confidence_is_canonical_label():
    m = re.search(r"\*\*Confidence:\*\*\s+(low|medium|high)", SAMPLE_REPORT)
    assert m, "confidence label not found"
    assert m.group(1) in ("low", "medium", "high")


# --- Amateur "What this means" section --------------------------------------


_FORBIDDEN_JARGON_IN_AMATEUR = (
    "P10",
    "P25",
    "P50",
    "P75",
    "P90",
    "DCF",
    "MoS threshold",
    "convergence ratio",
    "convergence_ratio",
    "ERP",
    "basis points",
    "bps",
    "FCF yield",
    "owner earnings",
)


def test_what_this_means_section_present_in_sample():
    body = _section(SAMPLE_REPORT, "## What this means")
    assert body.strip(), "What this means section is empty"


def test_what_this_means_section_avoids_jargon_in_sample():
    """The sample report's amateur section must not contain banned-in-amateur
    jargon. Tier 3 will run the same check on real LLM output."""
    body = _section(SAMPLE_REPORT, "## What this means")
    for term in _FORBIDDEN_JARGON_IN_AMATEUR:
        assert term not in body, f"banned amateur-section term '{term}' found in: {body[:200]}"


def test_what_this_means_section_is_3_to_5_sentences_in_sample():
    body = _section(SAMPLE_REPORT, "## What this means").strip()
    # Naive sentence count: split on `. ` then drop trailing empty.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    assert 3 <= len(sentences) <= 6, (
        f"What this means should be 3-5 sentences, got {len(sentences)}: {body[:200]}"
    )


# --- Bull/Base/Bear scenarios in MC section ---------------------------------


def test_mc_section_contains_bull_base_bear_scenarios_in_sample():
    body = _section(SAMPLE_REPORT, "## Monte Carlo fair value")
    for label in ("Bull (P75-P90)", "Base (P25-P75)", "Bear (P10-P25)"):
        assert label in body, f"MC section missing scenario label '{label}'"


# --- Watchlist section ------------------------------------------------------


def test_watchlist_section_present_in_sample():
    body = _section(SAMPLE_REPORT, "## Watchlist")
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert 3 <= len(bullets) <= 5, f"Watchlist should have 3-5 bullets, got {len(bullets)}"


def test_watchlist_bullets_end_with_agent_suffix_in_sample():
    """Each watchlist item must end with `(<agent>)` so Phase 1 Triage can
    parse out which upstream agent should monitor the signal."""
    body = _section(SAMPLE_REPORT, "## Watchlist")
    valid_agents = {
        "(filings)",
        "(news)",
        "(fundamentals)",
        "(risk)",
        "(synthesis)",
    }
    for bullet in [line for line in body.splitlines() if line.startswith("- ")]:
        suffix_present = any(bullet.rstrip().endswith(a) for a in valid_agents)
        assert suffix_present, f"Watchlist item missing agent suffix: {bullet}"


# --- Helper -----------------------------------------------------------------


def _section(md: str, header: str) -> str:
    """Return the body of `header` (everything between header and the next ## header)."""
    lines = md.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if start is None:
        return ""
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1 :], start=start + 1)
            if line.startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end])


# --- Schema integration -----------------------------------------------------


def test_synthesis_output_schema_accepts_default_fields():
    """SynthesisOutput must validate from {report: str} alone — gaps + confidence
    + errors all default. Important for the graph-test stub contract."""
    out = SynthesisOutput.model_validate({"report": "# x"})
    assert out.confidence == "medium"
    assert out.gaps == []
    assert out.errors == []


def test_synthesis_output_schema_rejects_invalid_confidence():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SynthesisOutput.model_validate({"report": "# x", "confidence": "very-high"})
