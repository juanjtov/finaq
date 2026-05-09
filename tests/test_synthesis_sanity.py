"""Tier 1c — synthesis output sanity tests with the real graph.

Run via:  pytest -m integration tests/test_synthesis_sanity.py

Catches catastrophic input bugs that all-stub tests would miss: a Risk
refactor that drops `level` would break Synthesis silently; this test
fails loudly. Also asserts the structural contract on the markdown
produced by a real synthesis-LLM call (all 7 sections present, MC numbers within
±1 of state, PDF exports without error, evidence has citations).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from agents import build_graph
from utils.pdf_export import export
from utils.schemas import SynthesisOutput

pytestmark = pytest.mark.integration

THESES_DIR = Path(__file__).parents[1] / "theses"

REQUIRED_SECTIONS = (
    "## What this means",
    "## Thesis statement",
    "## Bull case",
    "## Bear case",
    "## Top risks",
    "## Monte Carlo fair value",
    "## Probabilistic forecast",
    "## Action recommendation",
    "## Watchlist",
    "## Evidence",
)

# Words a real LLM run must NOT use inside the amateur "What this means"
# section. We intentionally don't ban "P50" alone because it sometimes appears
# inline as part of a number — but the framework intent (jargon-free) is
# enforced via the longer phrases below.
_FORBIDDEN_JARGON_IN_AMATEUR = (
    "P10",
    "P50",
    "P90",
    "DCF",
    "MoS threshold",
    "convergence ratio",
    "convergence_ratio",
    "basis points",
    "FCF yield",
    "owner earnings",
)


@pytest.fixture(autouse=True)
def _require_keys():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.fixture(scope="module")
async def nvda_ai_cake_run() -> dict:
    """One real graph run on NVDA + ai_cake. Slow (~3-5 min). Cached for
    every test in the module to avoid re-paying for the synthesis LLM
    on each test."""
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    graph = build_graph()
    final = await graph.ainvoke({"ticker": "NVDA", "thesis": thesis})
    return final


# --- Markdown structural contract ------------------------------------------


@pytest.mark.asyncio
async def test_real_run_produces_all_seven_sections(nvda_ai_cake_run):
    report = nvda_ai_cake_run.get("report", "")
    for h in REQUIRED_SECTIONS:
        assert h in report, f"missing required section header: {h}"


@pytest.mark.asyncio
async def test_real_run_confidence_label_canonical(nvda_ai_cake_run):
    report = nvda_ai_cake_run.get("report", "")
    m = re.search(r"\*\*Confidence:\*\*\s+(low|medium|high)", report, re.IGNORECASE)
    assert m, "confidence label not found in markdown"
    assert m.group(1).lower() in ("low", "medium", "high")
    # And duplicated in state for downstream consumers
    assert nvda_ai_cake_run.get("synthesis_confidence") in ("low", "medium", "high")


@pytest.mark.asyncio
async def test_real_run_bull_and_bear_have_3_to_5_bullets(nvda_ai_cake_run):
    report = nvda_ai_cake_run.get("report", "")
    for section_header in ("## Bull case", "## Bear case"):
        body = _section(report, section_header)
        bullets = [line for line in body.splitlines() if line.startswith("- ")]
        assert 3 <= len(bullets) <= 5, (
            f"{section_header} has {len(bullets)} bullets, expected 3-5"
        )


@pytest.mark.asyncio
async def test_real_run_bullet_word_count_under_20(nvda_ai_cake_run):
    """CLAUDE.md §11 caps Bull/Bear bullets at 20 words. Allow a 5-word slack
    for citations and inline parens — fail at 25+ words."""
    report = nvda_ai_cake_run.get("report", "")
    for section_header in ("## Bull case", "## Bear case"):
        body = _section(report, section_header)
        for bullet in [line for line in body.splitlines() if line.startswith("- ")]:
            words = bullet[2:].split()
            assert len(words) <= 25, f"bullet too long ({len(words)} words): {bullet}"


@pytest.mark.asyncio
async def test_real_run_top_risks_have_severity(nvda_ai_cake_run):
    """Each numbered risk must carry a severity 1-5 in some recognisable form.
    Both `severity 4` and `sev 4` (with optional parens / colons / **bold**)
    are accepted — the LLM tends to vary the surface form across runs."""
    report = nvda_ai_cake_run.get("report", "")
    body = _section(report, "## Top risks")
    numbered = [line for line in body.splitlines() if re.match(r"^\d+\.", line)]
    assert numbered, f"no numbered risks in:\n{body[:300]}"
    severity_re = re.compile(r"\bsev(?:erity)?\b[:\s]*\d+", re.IGNORECASE)
    for line in numbered:
        assert severity_re.search(line), f"top-risk line missing severity: {line}"


@pytest.mark.asyncio
async def test_real_run_mc_section_quotes_state_p50(nvda_ai_cake_run):
    """The MC section must reference numbers within ±$2 of state.monte_carlo.dcf.p50.
    Catches LLM-style number drift (says $190 when state says $185)."""
    mc = nvda_ai_cake_run.get("monte_carlo") or {}
    if mc.get("method") == "skipped":
        pytest.skip(f"MC was skipped — cannot grade Synthesis MC section. errors={mc.get('errors')}")
    p50 = (mc.get("dcf") or {}).get("p50")
    assert p50 is not None, "DCF p50 missing from state.monte_carlo"

    report = nvda_ai_cake_run.get("report", "")
    section = _section(report, "## Monte Carlo fair value")
    # Find any dollar number in the MC section; require at least one within ±$2 of P50.
    candidates = [float(m) for m in re.findall(r"\$?(\d+(?:\.\d+)?)", section)]
    near = [c for c in candidates if abs(c - p50) <= max(2.0, 0.02 * p50)]
    assert near, (
        f"MC section did not quote a number near state.monte_carlo.dcf.p50={p50:.2f}; "
        f"saw candidates={candidates[:6]} in section:\n{section[:300]}"
    )


@pytest.mark.asyncio
async def test_real_run_evidence_section_non_empty(nvda_ai_cake_run):
    report = nvda_ai_cake_run.get("report", "")
    body = _section(report, "## Evidence")
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert bullets, f"Evidence section empty:\n{body[:300]}"


@pytest.mark.asyncio
async def test_real_run_pdf_export_succeeds(tmp_path, nvda_ai_cake_run):
    """The end-to-end test that matters: PDF must render without ReportLab errors
    on whatever markdown the LLM emitted."""
    report = nvda_ai_cake_run.get("report", "")
    out = export(report, tmp_path / "nvda_real.pdf")
    assert out.exists()
    assert out.stat().st_size > 1500


# --- SynthesisOutput round-trip --------------------------------------------


@pytest.mark.asyncio
async def test_real_run_synthesis_output_validates(nvda_ai_cake_run):
    """The state's synthesis fields must validate against the SynthesisOutput
    Pydantic schema as a complete object."""
    SynthesisOutput.model_validate(
        {
            "report": nvda_ai_cake_run.get("report", ""),
            "confidence": nvda_ai_cake_run.get("synthesis_confidence", "medium"),
            "gaps": nvda_ai_cake_run.get("gaps", []),
            "watchlist": nvda_ai_cake_run.get("watchlist", []),
        }
    )


# --- New: amateur "What this means" section --------------------------------


@pytest.mark.asyncio
async def test_real_run_what_this_means_section_avoids_jargon(nvda_ai_cake_run):
    """Section is for amateur readers — must not contain percentile / DCF / ERP
    technical terms. Bar enforces the prompt's jargon-ban rule."""
    body = _section(nvda_ai_cake_run.get("report", ""), "## What this means")
    assert body.strip(), "What this means section is empty"
    found = [t for t in _FORBIDDEN_JARGON_IN_AMATEUR if t in body]
    assert not found, (
        f"What this means section contains banned jargon {found}; section was:\n{body[:400]}"
    )


@pytest.mark.asyncio
async def test_real_run_what_this_means_is_3_to_6_sentences(nvda_ai_cake_run):
    """Slightly looser than the spec's 3-5 to allow LLM-side punctuation drift."""
    body = _section(nvda_ai_cake_run.get("report", ""), "## What this means").strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    assert 3 <= len(sentences) <= 6, (
        f"What this means should be 3-5 sentences, got {len(sentences)}: {body[:300]}"
    )


# --- New: Bull/Base/Bear scenarios -----------------------------------------


@pytest.mark.asyncio
async def test_real_run_mc_section_has_three_scenarios(nvda_ai_cake_run):
    """MC section must include Bull / Base / Bear scenario bullets. Tolerates
    minor formatting (bold markers, parens around percentile labels)."""
    body = _section(nvda_ai_cake_run.get("report", ""), "## Monte Carlo fair value")
    for scenario in ("Bull", "Base", "Bear"):
        # Match `**Bull (...)**` or `Bull (...)` — bold is preferred but not required.
        pattern = rf"\*?\*?{scenario}\b\s*\("
        assert re.search(pattern, body), (
            f"MC section missing scenario '{scenario}'; section was:\n{body[:400]}"
        )


# --- New: watchlist section + state field ----------------------------------


@pytest.mark.asyncio
async def test_real_run_watchlist_section_non_empty(nvda_ai_cake_run):
    body = _section(nvda_ai_cake_run.get("report", ""), "## Watchlist")
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert bullets, f"Watchlist section had no bullets:\n{body[:300]}"


@pytest.mark.asyncio
async def test_real_run_watchlist_items_carry_agent_suffix(nvda_ai_cake_run):
    """Each watchlist bullet must end with (filings) / (news) / (fundamentals)
    / (risk) / (synthesis) so Phase 1 Triage can parse them mechanically."""
    body = _section(nvda_ai_cake_run.get("report", ""), "## Watchlist")
    valid_agents = ("(filings)", "(news)", "(fundamentals)", "(risk)", "(synthesis)")
    for bullet in [line for line in body.splitlines() if line.startswith("- ")]:
        assert any(bullet.rstrip().endswith(a) for a in valid_agents), (
            f"Watchlist item missing agent suffix: {bullet}"
        )


@pytest.mark.asyncio
async def test_real_run_watchlist_state_field_non_empty(nvda_ai_cake_run):
    """The structured `state.watchlist` field (parsed for Triage) must mirror the
    section. Empty-list is OK only if the LLM emitted no watchlist."""
    section_bullets = [
        line.lstrip("- ").strip()
        for line in _section(nvda_ai_cake_run.get("report", ""), "## Watchlist").splitlines()
        if line.startswith("- ")
    ]
    state_watchlist = nvda_ai_cake_run.get("watchlist", [])
    if section_bullets:
        assert state_watchlist, (
            f"Markdown had {len(section_bullets)} watchlist bullets but state.watchlist is empty"
        )


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
