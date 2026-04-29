"""Tier 2 Synthesis evaluation — LLM-as-judge for the qualities Tier 1 can't see.

Five categorical judges (NONE / WEAK / PARTIAL / HIGH), rationale-first JSON
ordering (the same anti-pattern fix we used in the RAG and Risk eval prompts):

  1. Faithfulness          Are quoted spans / numbers grounded in upstream evidence?
  2. Thesis-awareness      Does the report condition on the active thesis pivot?
  3. Tension handling      When upstream sources disagreed, does Synthesis acknowledge it?
  4. Action specificity    Is the recommendation specific (sized, conditional, threshold-based)?
  5. Confidence calibration  Does `confidence` match the agreement-pattern + risk.level?

Costs ~$0.005/judgement × 5 ≈ $0.025/run plus one full Synthesis call. Gated
`pytest -m eval`.

These tests run synthesis ONCE per test module via a session-scoped fixture
to avoid re-paying for the Opus call across each judge category.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.synthesis import run as synthesis_run
from utils import logger
from utils.models import MODEL_JUDGE
from utils.openrouter import get_client
from utils.rag_eval import write_eval_run
from utils.schemas import SynthesisOutput

pytestmark = pytest.mark.eval

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    judge = os.environ.get("MODEL_JUDGE", "")
    if not judge or judge == "test-stub-model":
        pytest.skip("MODEL_JUDGE not set in .env")


# --- Reusable upstream-state fixture (Risk pre-baked, no real workers) ------


def _stub_full_state() -> dict:
    """A schema-valid upstream state that Synthesis can write a full report from
    without us paying for Fundamentals/Filings/News/Risk LLM calls. Faithfully
    mirrors the shape of a real run so the judges have realistic content to grade."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    return {
        "ticker": "NVDA",
        "thesis": thesis,
        "fundamentals": {
            "summary": (
                "NVDA: 5-year revenue CAGR 100%; latest quarter +47% YoY to $60.9B. "
                "Operating margins 65%. FCF yield 1.84% — well below 4% MoS threshold."
            ),
            "kpis": {
                "revenue_5y_cagr": 1.0,
                "fcf_yield": 1.84,
                "operating_margin_5yr_avg": 0.49,
                "pe_trailing": 44.3,
                "current_price": 200.0,
                "shares_outstanding": 24e9,
                "revenue_latest": 60.9e9,
                "net_cash": 30e9,
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
                    "excerpt": "$60.9B",
                    "as_of": "2026-04-25",
                }
            ],
        },
        "filings": {
            "summary": (
                "10-K and 10-Qs cite supply concentration (TSM N3/N2 sole supplier of "
                "leading-edge nodes), capacity constraints in Blackwell ramp, and "
                "export-control regulatory risk on advanced GPUs."
            ),
            "risk_themes": [
                "supply concentration",
                "export-control regulation",
                "customer concentration",
            ],
            "mdna_quotes": [
                {
                    "text": "We continue to face supply constraints across leading-edge nodes.",
                    "accession": "0001045810-26-000123",
                    "item": "Item 7",
                },
                {
                    "text": "A small number of customers account for a substantial portion of our revenue.",
                    "accession": "0001045810-26-000123",
                    "item": "Item 1A",
                },
            ],
            "evidence": [
                {
                    "source": "edgar",
                    "accession": "0001045810-26-000123",
                    "item": "Item 7",
                    "as_of": "2026-02-21",
                }
            ],
        },
        "news": {
            "summary": (
                "Hyperscalers (MSFT, GOOGL) raised AI capex guidance to combined $320B+; "
                "valuation premium concerns from sell-side as multiple compresses."
            ),
            "catalysts": [
                {
                    "title": "MSFT raised AI capex guidance to $80B for FY26",
                    "summary": "Microsoft increased data-center capex by 30% YoY",
                    "sentiment": "bull",
                    "url": "https://example.com/msft-capex",
                    "as_of": "2026-04-15",
                },
                {
                    "title": "GOOGL announced $75B capex floor for AI infra",
                    "summary": "Alphabet committed to floor capex spending through 2027",
                    "sentiment": "bull",
                    "url": "https://example.com/googl-capex",
                    "as_of": "2026-04-12",
                },
            ],
            "concerns": [
                {
                    "title": "Sell-side flags AI valuation premium fading",
                    "summary": "Multiple analysts trim NVDA price targets",
                    "sentiment": "bear",
                    "url": "https://example.com/sell-side",
                    "as_of": "2026-04-19",
                }
            ],
            "evidence": [
                {
                    "source": "tavily",
                    "url": "https://example.com/msft-capex",
                    "as_of": "2026-04-15",
                }
            ],
        },
        "risk": {
            "level": "ELEVATED",
            "score_0_to_10": 6,
            "summary": (
                "Multiple convergent signals on supply concentration. FCF yield breach "
                "of 4% MoS threshold. Tension between hyperscaler capex strength and "
                "valuation premium concerns."
            ),
            "top_risks": [
                {
                    "title": "Supply concentration on TSM N3/N2",
                    "severity": 4,
                    "explanation": "Single-supplier risk on leading-edge nodes; flagged in 10-K and confirmed by industry capacity reports.",
                    "sources": ["fundamentals", "filings"],
                },
                {
                    "title": "FCF yield below MoS threshold",
                    "severity": 3,
                    "explanation": "1.84% vs 4% threshold — implies investors paying for growth, not earnings.",
                    "sources": ["fundamentals"],
                },
                {
                    "title": "Customer concentration",
                    "severity": 3,
                    "explanation": "3 hyperscalers represent >50% of revenue.",
                    "sources": ["filings"],
                },
            ],
            "convergent_signals": [
                {
                    "theme": "supply concentration",
                    "sources": ["fundamentals", "filings"],
                    "explanation": "Fundamentals show margin sensitivity to component costs; filings confirm TSM dependency.",
                }
            ],
            "threshold_breaches": [
                {
                    "signal": "fcf_yield",
                    "operator": "<",
                    "threshold_value": 4,
                    "observed_value": 1.84,
                    "explanation": "Below 4% margin-of-safety threshold.",
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
            "multiple": {"p10": 110.0, "p25": 145.0, "p50": 180.0, "p75": 215.0, "p90": 255.0},
        },
    }


@pytest.fixture(scope="module")
async def synthesis_result() -> dict:
    """Run Synthesis once for the whole module — Opus calls are slow + expensive.
    All 5 judge tests share this output."""
    state = _stub_full_state()
    result = await synthesis_run(state)
    return {"state": state, "result": result, "report": result["report"]}


# --- Judge prompt + helper --------------------------------------------------


def _judge(category_prompt: str, report_excerpt: str, context: str) -> tuple[str, str]:
    """Single LLM call. Returns (label, rationale). Label ∈ NONE|WEAK|PARTIAL|HIGH."""
    client = get_client()
    user = (
        f"REPORT EXCERPT:\n{report_excerpt[:6000]}\n\n"
        f"CONTEXT (upstream agent outputs that fed Synthesis):\n{context[:4000]}"
    )
    resp = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[
            {"role": "system", "content": category_prompt},
            {"role": "user", "content": user},
        ],
        max_tokens=300,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        nl = raw.find("\n")
        if nl > 0:
            raw = raw[nl + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
    try:
        data = json.loads(raw)
        label = str(data.get("label", "NONE")).strip().upper()
        rationale = str(data.get("rationale", ""))
        if label not in ("NONE", "WEAK", "PARTIAL", "HIGH"):
            label = "NONE"
        return label, rationale
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"[synthesis-judge] unparseable: {raw[:120]} ({e})")
        return "NONE", "judge response unparseable"


def _label_score(label: str) -> int:
    return {"NONE": 0, "WEAK": 1, "PARTIAL": 2, "HIGH": 3}.get(label, 0)


def _render_context(state: dict) -> str:
    """Compress upstream state into a string the judge can read in one pass."""
    fund = state.get("fundamentals", {})
    filings = state.get("filings", {})
    news = state.get("news", {})
    risk = state.get("risk", {})
    mc = state.get("monte_carlo", {})
    parts = [
        f"FUNDAMENTALS summary: {fund.get('summary', '')}",
        f"FUNDAMENTALS kpis: {json.dumps(fund.get('kpis', {}))[:600]}",
        f"FILINGS summary: {filings.get('summary', '')}",
        f"FILINGS risk_themes: {filings.get('risk_themes', [])}",
        f"FILINGS quotes: {[q.get('text', '')[:150] for q in filings.get('mdna_quotes', [])]}",
        f"NEWS summary: {news.get('summary', '')}",
        f"NEWS catalysts: {[c.get('title', '') for c in news.get('catalysts', [])]}",
        f"NEWS concerns: {[c.get('title', '') for c in news.get('concerns', [])]}",
        f"RISK level: {risk.get('level', '?')}",
        f"RISK top_risks: {[r.get('title', '') for r in risk.get('top_risks', [])]}",
        f"RISK convergent_signals: {[s.get('theme', '') for s in risk.get('convergent_signals', [])]}",
        f"MC method: {mc.get('method', '?')}",
        f"MC dcf p50: {(mc.get('dcf') or {}).get('p50')}",
        f"MC convergence: {mc.get('convergence_ratio')}",
        f"MC current_price: {mc.get('current_price')}",
    ]
    return "\n".join(parts)


# --- Judge prompts (one per category) ---------------------------------------

_FAITHFULNESS_PROMPT = """You are a faithfulness grader for a financial-research synthesis agent.

You will see a REPORT EXCERPT (markdown produced by the synthesis agent) and
the CONTEXT it was given (structured outputs from upstream agents). Decide:
do the numbers, quotes, and named events in the report appear in the context,
or did the synthesis agent fabricate them?

Labels:
  NONE     Multiple fabrications. Numbers / quotes / events that don't appear in context.
  WEAK     One or two ungrounded claims; majority is grounded.
  PARTIAL  Mostly grounded but vague paraphrasing makes some claims unverifiable.
  HIGH     Every numeric claim and named entity in the excerpt traces back to context.

Output STRICT JSON, rationale BEFORE label:

{"rationale": "<one short sentence>", "label": "<NONE|WEAK|PARTIAL|HIGH>"}
"""

_THESIS_AWARENESS_PROMPT = """You grade thesis-awareness of an investment-research synthesis agent.

The active thesis has a load-bearing variable (for AI cake, that's data-center
capex / hyperscaler buildout / AI compute supply). A thesis-aware report
frames its analysis around that pivot. A non-thesis-aware report would read
identically for any tech name.

Labels:
  NONE     The report is generic; could be a stock report on any name.
  WEAK     One token reference to the thesis name; analysis is generic.
  PARTIAL  At least one section ties to the thesis pivot; others are generic.
  HIGH     Bull, bear, and action recommendation all explicitly anchor to the thesis pivot.

Output STRICT JSON, rationale BEFORE label:

{"rationale": "<one short sentence>", "label": "<NONE|WEAK|PARTIAL|HIGH>"}
"""

_TENSION_PROMPT = """You grade tension-handling in a synthesis report.

Upstream agents may disagree (e.g., Fundamentals strong but Risk elevated; News
bullish but Filings cautious). A good synthesis report names those tensions
and resolves them; a bad one papers over them with bland language.

Labels:
  NONE     Real tensions in the upstream context were ignored entirely.
  WEAK     A vague "however / on the other hand" phrase, no real resolution.
  PARTIAL  At least one tension named, with partial resolution.
  HIGH     Tensions are named, sources attributed, and resolution is given (often temporal).

Output STRICT JSON, rationale BEFORE label:

{"rationale": "<one short sentence>", "label": "<NONE|WEAK|PARTIAL|HIGH>"}
"""

_ACTION_SPECIFICITY_PROMPT = """You grade the action recommendation in a synthesis report.

The action recommendation is the user's reason to act or hold. Specific = sized
+ conditional + threshold-based ("Trim 20% if Q3 misses $42B guide; add 2% on
dip below $180"). Vague = "monitor / hold / consider trimming."

Labels:
  NONE     "No action" / "monitor" / "hold." Boilerplate.
  WEAK     A direction (trim/add) with no size and no condition.
  PARTIAL  A direction with one of {size, condition, threshold} but not all.
  HIGH     Direction, sizing, and a specific condition or threshold.

Output STRICT JSON, rationale BEFORE label:

{"rationale": "<one short sentence>", "label": "<NONE|WEAK|PARTIAL|HIGH>"}
"""

_CALIBRATION_PROMPT = """You grade confidence calibration in a synthesis report.

The report states a Confidence label (low / medium / high). Decide whether
that label matches the upstream agreement pattern and risk.level in CONTEXT:

  - HIGH should require ≥3 of 4 worker agents converging, MC convergence_ratio
    ≥ 0.7, risk.level ≤ ELEVATED.
  - LOW should appear when agents contradict OR multiple agents failed OR
    risk.level ≥ HIGH OR convergence_ratio < 0.4.
  - MEDIUM is the default for mixed-but-coherent signals.

Labels (about the calibration, not the report quality):
  NONE     Confidence is wildly miscalibrated (e.g., HIGH on contradictory upstream).
  WEAK     Confidence is one notch off in the wrong direction.
  PARTIAL  Confidence is defensible but weakly justified by the upstream pattern.
  HIGH     Confidence is well-calibrated to upstream agreement + risk.level.

Output STRICT JSON, rationale BEFORE label:

{"rationale": "<one short sentence>", "label": "<NONE|WEAK|PARTIAL|HIGH>"}
"""


def _section(md: str, header: str) -> str:
    """Extract one section's body for focused grading."""
    lines = md.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(header))
    except StopIteration:
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


# --- The five Tier-2 tests --------------------------------------------------


@pytest.mark.asyncio
async def test_faithfulness_judge(synthesis_result):
    """Bull + Bear + Top-risks combined, judged for grounded numbers/quotes."""
    md = synthesis_result["report"]
    excerpt = (
        f"## Bull case\n{_section(md, '## Bull case')}\n\n"
        f"## Bear case\n{_section(md, '## Bear case')}\n\n"
        f"## Top risks\n{_section(md, '## Top risks')}"
    )
    context = _render_context(synthesis_result["state"])
    label, rationale = _judge(_FAITHFULNESS_PROMPT, excerpt, context)
    score = _label_score(label)
    write_eval_run(
        {
            "tier": 2,
            "suite": "synthesis_faithfulness",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "label": label,
            "score": score,
            "rationale": rationale,
        }
    )
    # Bar: PARTIAL or higher (≥2). Allows for paraphrasing without crashing on a
    # single un-cited bullet, but rejects multi-fabrication.
    assert score >= 2, f"faithfulness too low: {label} — {rationale}"


@pytest.mark.asyncio
async def test_thesis_awareness_judge(synthesis_result):
    md = synthesis_result["report"]
    excerpt = (
        f"## Thesis statement\n{_section(md, '## Thesis statement')}\n\n"
        f"## Bull case\n{_section(md, '## Bull case')}\n\n"
        f"## Action recommendation\n{_section(md, '## Action recommendation')}"
    )
    context = _render_context(synthesis_result["state"])
    label, rationale = _judge(_THESIS_AWARENESS_PROMPT, excerpt, context)
    score = _label_score(label)
    write_eval_run(
        {
            "tier": 2,
            "suite": "synthesis_thesis_awareness",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "label": label,
            "score": score,
            "rationale": rationale,
        }
    )
    assert score >= 2, f"thesis-awareness too low: {label} — {rationale}"


@pytest.mark.asyncio
async def test_tension_handling_judge(synthesis_result):
    """Synthesis was given an explicit tension (hyperscaler capex bull vs valuation
    premium bear). Verify the report names + resolves it."""
    md = synthesis_result["report"]
    excerpt = (
        f"## Bull case\n{_section(md, '## Bull case')}\n\n"
        f"## Bear case\n{_section(md, '## Bear case')}\n\n"
        f"## Action recommendation\n{_section(md, '## Action recommendation')}"
    )
    context = _render_context(synthesis_result["state"])
    label, rationale = _judge(_TENSION_PROMPT, excerpt, context)
    score = _label_score(label)
    write_eval_run(
        {
            "tier": 2,
            "suite": "synthesis_tension_handling",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "label": label,
            "score": score,
            "rationale": rationale,
        }
    )
    # Bar slightly looser (≥1) — papering over is bad but tension-naming is the
    # hardest of the 5 judges and the model may be terse.
    assert score >= 1, f"tension handling too low: {label} — {rationale}"


@pytest.mark.asyncio
async def test_action_specificity_judge(synthesis_result):
    md = synthesis_result["report"]
    excerpt = _section(md, "## Action recommendation")
    context = _render_context(synthesis_result["state"])
    label, rationale = _judge(_ACTION_SPECIFICITY_PROMPT, excerpt, context)
    score = _label_score(label)
    write_eval_run(
        {
            "tier": 2,
            "suite": "synthesis_action_specificity",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "label": label,
            "score": score,
            "rationale": rationale,
        }
    )
    assert score >= 2, f"action recommendation too vague: {label} — {rationale}"


@pytest.mark.asyncio
async def test_confidence_calibration_judge(synthesis_result):
    md = synthesis_result["report"]
    confidence_line = next(
        (line for line in md.splitlines() if "Confidence:" in line),
        "Confidence: (not found)",
    )
    excerpt = (
        f"{confidence_line}\n\n"
        f"## Top risks\n{_section(md, '## Top risks')}\n\n"
        f"## Monte Carlo fair value\n{_section(md, '## Monte Carlo fair value')}"
    )
    context = _render_context(synthesis_result["state"])
    label, rationale = _judge(_CALIBRATION_PROMPT, excerpt, context)
    score = _label_score(label)
    write_eval_run(
        {
            "tier": 2,
            "suite": "synthesis_confidence_calibration",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "synthesis_confidence": synthesis_result["result"]["synthesis_confidence"],
            "label": label,
            "score": score,
            "rationale": rationale,
        }
    )
    # Bar: PARTIAL — calibration is judgment-heavy; HIGH is rare.
    assert score >= 2, f"confidence miscalibrated: {label} — {rationale}"


# --- Aggregate summary row (Mission Control) -------------------------------


@pytest.mark.asyncio
async def test_synthesis_eval_summary_recorded(synthesis_result):
    """One run, one row — summary of the markdown shape + the SynthesisOutput
    fields. Tier 2 judges already wrote per-category rows; this is the
    one-row-per-Synthesis-call summary."""
    out = SynthesisOutput.model_validate(
        {
            "report": synthesis_result["report"],
            "confidence": synthesis_result["result"]["synthesis_confidence"],
            "gaps": synthesis_result["result"]["gaps"],
        }
    )
    write_eval_run(
        {
            "tier": 2,
            "suite": "synthesis_summary",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "confidence": out.confidence,
            "gaps_count": len(out.gaps),
            "report_chars": len(out.report),
        }
    )
    assert out.report
