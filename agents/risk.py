"""Risk agent — synthesis-only cross-modal red-flag synthesizer.

Reads `state.fundamentals`, `state.filings`, `state.news` plus `state.thesis`.
Does NOT call external services. One LLM call (configured via MODEL_RISK)
classifies overall risk into a categorical level (LOW..CRITICAL) and emits
top_risks + convergent_signals + threshold_breaches.

The composite `score_0_to_10` is *derived* from `level` via
`RISK_LEVEL_TO_SCORE` — the LLM never picks an integer directly. This sidesteps
the unstable-numeric-scale problem we hit with the LLM judge prompt.

Risk does NOT directly modify Monte Carlo inputs in Phase 0 — see
docs/ARCHITECTURE.md §6.10 (Approach A). Synthesis composes Risk's `level` and
MC's P10/P50/P90 side-by-side in the final report.

Run standalone:  python -m agents.risk NVDA [thesis_slug]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from utils import logger
from utils.models import MODEL_RISK
from utils.openrouter import get_client
from utils.schemas import RISK_LEVEL_TO_SCORE, RiskOutput
from utils.state import FinaqState

NODE = "risk"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "risk.md").read_text()
LLM_MAX_TOKENS = 2500

# Conservative fallback when the LLM call fails — keeps the graph alive but
# clearly flags that the risk view is degraded.
FALLBACK_RISK = {
    "level": "MODERATE",
    "score_0_to_10": RISK_LEVEL_TO_SCORE["MODERATE"],
    "top_risks": [
        {
            "title": "[fallback] Risk synthesis failed; treat with caution",
            "severity": 3,
            "explanation": "The Risk agent could not synthesize across worker outputs. Read the individual Fundamentals / Filings / News sections directly.",
            "sources": [],
        }
    ],
    "convergent_signals": [],
    "threshold_breaches": [],
    "summary": "Risk synthesis unavailable — see individual worker outputs.",
}


# --- Prompt assembly --------------------------------------------------------


def _summarise_worker(name: str, payload: dict) -> str:
    """Render one worker's structured output as a compact prompt section."""
    if not payload:
        return f"\n=== {name.upper()} ===\n(no output — agent did not run or failed)"

    parts = [f"\n=== {name.upper()} ==="]
    if "summary" in payload:
        parts.append(f"summary: {payload.get('summary', '')}")

    # Per-agent specifics
    if name == "fundamentals":
        kpis = payload.get("kpis") or {}
        if kpis:
            parts.append("kpis:")
            parts.append(json.dumps(kpis, indent=2, default=str))
        proj = payload.get("projections") or {}
        if proj:
            parts.append(f"projections: {json.dumps(proj, default=str)}")
    elif name == "filings":
        risks = payload.get("risk_themes") or []
        if risks:
            parts.append(f"risk_themes: {risks}")
        quotes = payload.get("mdna_quotes") or []
        if quotes:
            parts.append(f"mdna_quotes ({len(quotes)}):")
            for q in quotes[:5]:  # cap to keep prompt size sane
                parts.append(f"  - [{q.get('item', '?')}] {q.get('text', '')[:200]}")
    elif name == "news":
        cats = payload.get("catalysts") or []
        if cats:
            parts.append(f"catalysts ({len(cats)}):")
            for c in cats:
                parts.append(
                    f"  - [{c.get('sentiment', '?')}] {c.get('title', '')} "
                    f"({c.get('as_of', 'unknown')})"
                )
        concerns = payload.get("concerns") or []
        if concerns:
            parts.append(f"concerns ({len(concerns)}):")
            for c in concerns:
                parts.append(
                    f"  - [{c.get('sentiment', '?')}] {c.get('title', '')} "
                    f"({c.get('as_of', 'unknown')})"
                )

    return "\n".join(parts)


def _build_user_prompt(ticker: str, thesis: dict, state: FinaqState) -> str:
    parts = [
        f"TICKER: {ticker}",
        f"ACTIVE THESIS: {thesis.get('name', 'unknown')}",
        f"THESIS SUMMARY: {thesis.get('summary', '')}",
        f"ANCHOR TICKERS: {', '.join(thesis.get('anchor_tickers', []))}",
        "",
        "MATERIAL THRESHOLDS THE THESIS WATCHES:",
        json.dumps(thesis.get("material_thresholds", []), indent=2),
        "",
        "WORKER AGENT OUTPUTS:",
        _summarise_worker("fundamentals", state.get("fundamentals") or {}),
        _summarise_worker("filings", state.get("filings") or {}),
        _summarise_worker("news", state.get("news") or {}),
        "",
        "PRODUCE YOUR ANALYSIS NOW. STRICT JSON ONLY.",
    ]
    return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text


def _coerce_to_risk_output(raw: dict) -> dict:
    """Normalise the LLM's raw JSON into RiskOutput-shape.

    The LLM picks a `level`; we *derive* `score_0_to_10` to guarantee schema
    consistency (the model_validator on RiskOutput would reject any mismatch).
    """
    level = str(raw.get("level", "MODERATE")).strip().upper()
    if level not in RISK_LEVEL_TO_SCORE:
        logger.warning(f"[risk] LLM returned unknown level {level!r}; defaulting to MODERATE")
        level = "MODERATE"
    return {
        "level": level,
        "score_0_to_10": RISK_LEVEL_TO_SCORE[level],
        "summary": str(raw.get("summary", "")),
        "top_risks": raw.get("top_risks", []),
        "convergent_signals": raw.get("convergent_signals", []),
        "threshold_breaches": raw.get("threshold_breaches", []),
    }


def _call_llm(ticker: str, thesis: dict, state: FinaqState) -> dict:
    client = get_client()
    user = _build_user_prompt(ticker, thesis, state)
    resp = client.chat.completions.create(
        model=MODEL_RISK,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(_strip_code_fences(raw))
    return _coerce_to_risk_output(parsed)


# --- Graph node -------------------------------------------------------------


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    ticker = state.get("ticker", "")
    thesis = state.get("thesis") or {}
    errors: list[str] = []

    # Risk has no external dependencies — only reads other workers' state.
    # If all three workers failed to produce anything, the synthesis is meaningless.
    has_any_input = any(bool(state.get(n)) for n in ("fundamentals", "filings", "news"))
    if not has_any_input:
        out = RiskOutput.model_validate(
            {
                **FALLBACK_RISK,
                "summary": "No worker outputs available; risk view degraded.",
            }
        )
        out.errors = ["no upstream worker outputs"]
    else:
        try:
            llm_out = await asyncio.to_thread(_call_llm, ticker, thesis, state)
            out = RiskOutput.model_validate(llm_out)
            out.errors = errors
        except Exception as e:
            logger.error(f"[risk] LLM synthesis failed for {ticker}: {e}")
            errors.append(f"llm: {e}")
            out = RiskOutput.model_validate({**FALLBACK_RISK, "errors": errors})

    return {
        "risk": out.model_dump(),
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }


# --- CLI --------------------------------------------------------------------


async def _cli(ticker: str, thesis_slug: str = "ai_cake") -> None:
    """Standalone CLI: runs Fundamentals + Filings + News once to populate
    state, then runs Risk on the combined output. Slow (~1 min)."""
    from agents.filings import run as filings_run
    from agents.fundamentals import run as fundamentals_run
    from agents.news import run as news_run

    thesis = json.loads(Path(f"theses/{thesis_slug}.json").read_text())
    state: FinaqState = {"ticker": ticker, "thesis": thesis}

    # Run the three workers (sequentially for the CLI — graph runs them parallel)
    for worker_name, worker_run in (
        ("fundamentals", fundamentals_run),
        ("filings", filings_run),
        ("news", news_run),
    ):
        result = await worker_run(state)
        state.update({k: v for k, v in result.items() if k != "messages"})  # type: ignore[typeddict-item]
        print(f"[cli] {worker_name} done", file=sys.stderr)

    risk_result = await run(state)
    print(json.dumps(risk_result["risk"], indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agents.risk TICKER [thesis_slug]", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()
    thesis_slug = sys.argv[2] if len(sys.argv) > 2 else "ai_cake"
    asyncio.run(_cli(ticker, thesis_slug))
