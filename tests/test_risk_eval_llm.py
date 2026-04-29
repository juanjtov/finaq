"""Tier 2 Risk evaluation — LLM-as-judge for top_risks groundedness + severity sanity.

For each top_risk surfaced by the Risk agent, the judge model decides whether
the risk is genuinely supported by the worker outputs. Computes:

  - groundedness_rate  (fraction of top_risks the judge calls supported)
  - severity_sanity    (fraction with severity that matches the judge's read)

Costs ~$0.005/risk × ~5 risks ≈ $0.025/run. Gated `pytest -m eval`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from agents.risk import run
from tests.test_risk_quality import _stub_state
from utils import logger
from utils.models import MODEL_JUDGE
from utils.openrouter import get_client
from utils.rag_eval import write_eval_run
from utils.schemas import RiskOutput

pytestmark = pytest.mark.eval

THESES_DIR = Path(__file__).parents[1] / "theses"


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")
    judge = os.environ.get("MODEL_JUDGE", "")
    if not judge or judge == "test-stub-model":
        pytest.skip("MODEL_JUDGE not set in .env")


_RISK_JUDGE_PROMPT = """You are a quality grader for a risk-synthesis agent.

You will be given the COMBINED OUTPUTS of three worker agents (Fundamentals,
Filings, News) for a single ticker, plus ONE risk that another agent flagged.
Decide:
  1. Is the risk genuinely supported by content in the worker outputs?
     (Or is it speculative / fabricated?)
  2. Is the severity (1-5) reasonable given the evidence?

Output STRICT JSON, rationale BEFORE labels (think first, commit after):

{"rationale": "<one short sentence>", "supported": "SUPPORTED|UNSUPPORTED", "severity_judgment": "REASONABLE|TOO_HIGH|TOO_LOW|UNCLEAR"}
"""


@dataclass
class RiskItemJudgement:
    item_index: int
    supported: bool
    severity_judgment: str  # REASONABLE | TOO_HIGH | TOO_LOW | UNCLEAR
    rationale: str


def _judge_one_risk(workers_summary: str, risk: dict) -> RiskItemJudgement:
    client = get_client()
    user = (
        f"WORKER OUTPUTS:\n{workers_summary}\n\n"
        f"RISK BEING JUDGED:\n"
        f"  title:       {risk.get('title', '')}\n"
        f"  severity:    {risk.get('severity', '?')}\n"
        f"  explanation: {risk.get('explanation', '')}\n"
        f"  sources:     {risk.get('sources', [])}\n"
    )
    resp = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[
            {"role": "system", "content": _RISK_JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=200,
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
        supported = str(data.get("supported", "")).upper() == "SUPPORTED"
        severity_judgment = str(data.get("severity_judgment", "UNCLEAR")).upper()
        rationale = str(data.get("rationale", ""))
        return RiskItemJudgement(-1, supported, severity_judgment, rationale)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"[risk-judge] unparseable: {raw[:120]} ({e})")
        return RiskItemJudgement(-1, False, "UNCLEAR", "judge response unparseable")


@pytest.mark.asyncio
async def test_risk_top_risks_grounded_in_worker_outputs():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    state = _stub_state("NVDA", thesis)
    result = await run(state)
    out = RiskOutput.model_validate(result["risk"])

    if not out.top_risks:
        pytest.skip("No top_risks emitted")

    # Build a single concatenated workers-summary string for the judge.
    workers_summary = (
        f"FUNDAMENTALS:\n{state['fundamentals'].get('summary', '')}\n"
        f"  kpis: {json.dumps(state['fundamentals'].get('kpis', {}))}\n\n"
        f"FILINGS:\n{state['filings'].get('summary', '')}\n"
        f"  risk_themes: {state['filings'].get('risk_themes', [])}\n\n"
        f"NEWS:\n{state['news'].get('summary', '')}\n"
        f"  catalysts: {[c['title'] for c in state['news'].get('catalysts', [])]}\n"
        f"  concerns: {[c['title'] for c in state['news'].get('concerns', [])]}"
    )

    judgements: list[dict] = []
    supported_count = 0
    severity_reasonable_count = 0
    severity_decided_count = 0

    for i, r in enumerate(out.top_risks):
        j = _judge_one_risk(workers_summary, r.model_dump())
        if j.supported:
            supported_count += 1
        if j.severity_judgment in ("REASONABLE", "TOO_HIGH", "TOO_LOW"):
            severity_decided_count += 1
            if j.severity_judgment == "REASONABLE":
                severity_reasonable_count += 1
        judgements.append(asdict(j) | {"item_index": i, "title": r.title, "severity": r.severity})

    n = len(out.top_risks)
    groundedness_rate = supported_count / n
    severity_sanity = (
        severity_reasonable_count / severity_decided_count if severity_decided_count else 1.0
    )

    write_eval_run(
        {
            "tier": 2,
            "suite": "risk_llm_judge",
            "ticker": "NVDA",
            "thesis": "ai_cake",
            "level": out.level,
            "top_risks_total": n,
            "groundedness_rate": groundedness_rate,
            "severity_sanity": severity_sanity,
            "judgements": judgements,
        }
    )

    # Bars (calibrated for stochastic judge):
    #  - 70% of top_risks should be supported (most are real, not speculative)
    #  - 60% of severities should be reasonable (a bit looser; severity is fuzzier)
    assert groundedness_rate >= 0.7, (
        f"top_risks groundedness below 70%: {groundedness_rate:.2%} " f"({supported_count}/{n})"
    )
    assert severity_sanity >= 0.6, f"severity sanity below 60%: {severity_sanity:.2%}"
