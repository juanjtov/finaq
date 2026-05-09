"""Synthesis agent — final report writer (Step 7).

Single LLM call (configured via MODEL_SYNTHESIS) that integrates all four
upstream agent outputs plus the Monte Carlo distribution into a markdown
report following CLAUDE.md §11 exactly.

The agent is the *only* place in the graph that produces user-facing prose.
Everything upstream is structured data; everything downstream (PDF export,
Streamlit, Telegram) is rendering. So Synthesis is the system's narrative
bottleneck — see docs/ARCHITECTURE.md §6.18 for spine-mapping rationale.

Run standalone:  python -m agents.synthesis NVDA [thesis_slug]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

from utils import logger
from utils.models import MODEL_SYNTHESIS
from utils.openrouter import get_client
from utils.schemas import SynthesisOutput
from utils.state import FinaqState

NODE = "synthesis"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "synthesis.md").read_text()
LLM_MAX_TOKENS = 4000

# Per-section prompt-context caps. Synthesis takes the most complete view of
# state across the whole pipeline; without these caps, prompt size scales with
# upstream verbosity (10-K MD&A snippets, full news article excerpts) and we
# blow through the context window. These bounds keep the prompt under ~12k
# tokens for typical large-caps.
MAX_FILINGS_QUOTES = 6
MAX_NEWS_ITEMS = 6
MAX_TOP_RISKS = 7
MAX_EVIDENCE_PER_AGENT = 6
TEXT_FIELD_CHARS = 600


# --- Section formatters ------------------------------------------------------


def _truncate(text: str, n: int = TEXT_FIELD_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _format_fundamentals(payload: dict) -> str:
    if not payload:
        return "(fundamentals: agent did not produce output)"
    parts = ["### FUNDAMENTALS"]
    parts.append(f"summary: {_truncate(payload.get('summary', ''))}")
    kpis = payload.get("kpis") or {}
    if kpis:
        parts.append("kpis:")
        parts.append(json.dumps(kpis, indent=2, default=str))
    proj = payload.get("projections") or {}
    if proj:
        parts.append(f"projections: {json.dumps(proj, default=str)}")
    errs = payload.get("errors") or []
    if errs:
        parts.append(f"errors: {errs}")
    return "\n".join(parts)


def _format_filings(payload: dict) -> str:
    if not payload:
        return "(filings: agent did not produce output)"
    parts = ["### FILINGS"]
    parts.append(f"summary: {_truncate(payload.get('summary', ''))}")
    risk_themes = payload.get("risk_themes") or []
    if risk_themes:
        parts.append(f"risk_themes: {risk_themes}")
    quotes = payload.get("mdna_quotes") or []
    if quotes:
        parts.append(f"mdna_quotes ({len(quotes)}, showing first {MAX_FILINGS_QUOTES}):")
        for q in quotes[:MAX_FILINGS_QUOTES]:
            item = q.get("item", "?")
            accession = q.get("accession", "?")
            text = _truncate(q.get("text", ""), 350)
            parts.append(f'  - [item={item} accession={accession}] "{text}"')
    errs = payload.get("errors") or []
    if errs:
        parts.append(f"errors: {errs}")
    return "\n".join(parts)


def _format_news(payload: dict) -> str:
    if not payload:
        return "(news: agent did not produce output)"
    parts = ["### NEWS"]
    parts.append(f"summary: {_truncate(payload.get('summary', ''))}")
    cats = payload.get("catalysts") or []
    if cats:
        parts.append(f"catalysts ({len(cats)}, showing first {MAX_NEWS_ITEMS}):")
        for c in cats[:MAX_NEWS_ITEMS]:
            sent = c.get("sentiment", "?")
            title = c.get("title", "")
            url = c.get("url", "")
            as_of = c.get("as_of", "?")
            parts.append(f"  - [{sent} {as_of}] {title} ({url})")
    concerns = payload.get("concerns") or []
    if concerns:
        parts.append(f"concerns ({len(concerns)}, showing first {MAX_NEWS_ITEMS}):")
        for c in concerns[:MAX_NEWS_ITEMS]:
            sent = c.get("sentiment", "?")
            title = c.get("title", "")
            url = c.get("url", "")
            as_of = c.get("as_of", "?")
            parts.append(f"  - [{sent} {as_of}] {title} ({url})")
    errs = payload.get("errors") or []
    if errs:
        parts.append(f"errors: {errs}")
    return "\n".join(parts)


def _format_risk(payload: dict) -> str:
    if not payload:
        return "(risk: agent did not produce output)"
    parts = ["### RISK"]
    parts.append(f"level: {payload.get('level', '?')} (score {payload.get('score_0_to_10', '?')})")
    parts.append(f"summary: {_truncate(payload.get('summary', ''))}")
    top = payload.get("top_risks") or []
    if top:
        parts.append(f"top_risks ({len(top)}, showing first {MAX_TOP_RISKS}):")
        for t in top[:MAX_TOP_RISKS]:
            sev = t.get("severity", "?")
            title = t.get("title", "")
            sources = t.get("sources", [])
            expl = _truncate(t.get("explanation", ""), 250)
            parts.append(f"  - [sev={sev} sources={sources}] {title} — {expl}")
    cs = payload.get("convergent_signals") or []
    if cs:
        parts.append(f"convergent_signals ({len(cs)}):")
        for s in cs:
            theme = s.get("theme", "")
            sources = s.get("sources", [])
            expl = _truncate(s.get("explanation", ""), 250)
            parts.append(f"  - [{sources}] {theme} — {expl}")
    breaches = payload.get("threshold_breaches") or []
    if breaches:
        parts.append(f"threshold_breaches ({len(breaches)}):")
        for b in breaches:
            signal = b.get("signal", "")
            op = b.get("operator", "")
            tv = b.get("threshold_value", "")
            ov = b.get("observed_value", "")
            source = b.get("source", "")
            parts.append(f"  - {signal} {op} {tv} (observed={ov}, source={source})")
    return "\n".join(parts)


def _format_monte_carlo(payload: dict) -> str:
    if not payload or payload.get("method") in (None, "skipped"):
        errs = (payload or {}).get("errors") or []
        return f"### MONTE CARLO\n(skipped — {errs})"
    parts = ["### MONTE CARLO"]
    parts.append(f"method: {payload.get('method', '?')}")
    parts.append(f"current_price: {payload.get('current_price', '?')}")
    parts.append(f"discount_rate_used: {payload.get('discount_rate_used', '?'):.4f}")
    parts.append(f"terminal_growth_used: {payload.get('terminal_growth_used', '?')}")
    parts.append(f"convergence_ratio: {payload.get('convergence_ratio', '?')}")
    parts.append(f"n_sims: {payload.get('n_sims', '?')}, n_years: {payload.get('n_years', '?')}")
    dcf = payload.get("dcf") or {}
    if dcf:
        parts.append(
            f"dcf: P10={dcf.get('p10', '?'):.2f} "
            f"P25={dcf.get('p25', '?'):.2f} "
            f"P50={dcf.get('p50', '?'):.2f} "
            f"P75={dcf.get('p75', '?'):.2f} "
            f"P90={dcf.get('p90', '?'):.2f}"
        )
    mult = payload.get("multiple") or {}
    if mult:
        parts.append(
            f"multiple: P10={mult.get('p10', '?'):.2f} "
            f"P25={mult.get('p25', '?'):.2f} "
            f"P50={mult.get('p50', '?'):.2f} "
            f"P75={mult.get('p75', '?'):.2f} "
            f"P90={mult.get('p90', '?'):.2f}"
        )
    return "\n".join(parts)


def _collect_evidence(state: FinaqState) -> list[dict]:
    """Union of upstream agents' evidence lists, capped per agent. The LLM
    sees this list and is expected to copy the relevant items into the report's
    Evidence section. We don't fully de-duplicate (cheap to leave to the LLM)."""
    out: list[dict] = []
    for key in ("fundamentals", "filings", "news"):
        ev = (state.get(key) or {}).get("evidence") or []
        for e in ev[:MAX_EVIDENCE_PER_AGENT]:
            out.append({"agent": key, **e})
    return out


# --- Prompt assembly ---------------------------------------------------------


def _build_user_prompt(state: FinaqState) -> str:
    today_iso = date.today().isoformat()
    ticker = state.get("ticker", "?")
    thesis = state.get("thesis") or {}
    parts = [
        f"AS OF: {today_iso}",
        f"TICKER: {ticker}",
        f"ACTIVE THESIS: {thesis.get('name', 'unknown')}",
        f"THESIS SUMMARY: {thesis.get('summary', '')}",
        f"ANCHOR TICKERS: {', '.join(thesis.get('anchor_tickers', []))}",
        "",
        "MATERIAL THRESHOLDS THE THESIS WATCHES:",
        json.dumps(thesis.get("material_thresholds", []), indent=2),
        "",
        "## UPSTREAM AGENT OUTPUTS",
        "",
        _format_fundamentals(state.get("fundamentals") or {}),
        "",
        _format_filings(state.get("filings") or {}),
        "",
        _format_news(state.get("news") or {}),
        "",
        _format_risk(state.get("risk") or {}),
        "",
        _format_monte_carlo(state.get("monte_carlo") or {}),
        "",
        "## EVIDENCE INVENTORY (copy relevant items into the report's Evidence section)",
        json.dumps(_collect_evidence(state), indent=2, default=str)[:6000],
        "",
        f"PRODUCE THE REPORT NOW. Today's date for the report header: {today_iso}.",
        "STRICT JSON ONLY — keys: report, confidence, gaps.",
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


def _call_llm(state: FinaqState) -> dict:
    from utils.as_of import maybe_inject_as_of

    client = get_client()
    user = _build_user_prompt(state)
    system = maybe_inject_as_of(SYSTEM_PROMPT, state.get("as_of_date"))
    resp = client.chat.completions.create(
        model=MODEL_SYNTHESIS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(_strip_code_fences(raw))


def _coerce_string_list(raw: object) -> list[str]:
    """Coerce a possibly-malformed list-of-strings field into a clean list[str].
    Drops Nones, empty strings, and non-list inputs."""
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x]


def _extract_watchlist_from_markdown(report: str) -> list[str]:
    """Recover the Watchlist bullets from the markdown body. Used as a
    fallback when the LLM filled the `## Watchlist` section in the markdown
    but forgot to populate the JSON `watchlist` array — a real failure mode
    we observed during Step 7 prompt iteration."""
    lines = report.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("## Watchlist"))
    except StopIteration:
        return []
    bullets: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):  # next section
            break
        if line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets


def _coerce_to_synthesis_output(raw: dict) -> dict:
    """Normalise the LLM JSON into SynthesisOutput shape.

    Tolerates a missing `confidence` field (defaults to 'medium'); rejects
    missing or empty `report`. `gaps` and `watchlist` default to empty lists
    when missing; non-list values are coerced to []. If `watchlist` is empty
    but the markdown has a `## Watchlist` section with bullets, recover from
    the markdown — Phase 1 Triage relies on the structured field.
    """
    report = raw.get("report") or ""
    if not isinstance(report, str) or not report.strip():
        raise ValueError("Synthesis LLM returned empty `report`")
    confidence = str(raw.get("confidence", "medium")).strip().lower()
    if confidence not in ("low", "medium", "high"):
        logger.warning(f"[synthesis] LLM returned unknown confidence {confidence!r}; → medium")
        confidence = "medium"
    watchlist = _coerce_string_list(raw.get("watchlist"))
    if not watchlist:
        recovered = _extract_watchlist_from_markdown(report)
        if recovered:
            logger.info(
                f"[synthesis] watchlist JSON field empty; recovered {len(recovered)} "
                f"items from markdown section"
            )
            watchlist = recovered
    return {
        "report": report,
        "confidence": confidence,
        "gaps": _coerce_string_list(raw.get("gaps")),
        "watchlist": watchlist,
    }


# --- Fallback report (when LLM call fails) ----------------------------------


_FALLBACK_TEMPLATE = """# {ticker} — {thesis_name} thesis update

**Date:** {today} · **Confidence:** low

## What this means
The system could not write a final summary for this stock today. The raw data from each step is still available in the dashboard, but the integrated view is missing. Treat anything derived from this report with caution until the underlying issue is fixed. No action recommended.

## Thesis statement
[Synthesis unavailable — the final integration step failed. Read the upstream agent outputs directly via the Mission Control panel. The structured outputs are still valid.]

## Bull case
- [Synthesis failed; no bull integration available]

## Bear case
- [Synthesis failed; no bear integration available]

## Top risks
1. Synthesis failure itself — severity 3 — the final report could not be produced; treat all downstream actions with caution until the issue is resolved.

## Monte Carlo fair value
[See `state.monte_carlo` directly — Synthesis was not able to write this section.]

- **Bull (P75-P90):** unavailable — Synthesis failed.
- **Base (P25-P75):** unavailable — Synthesis failed.
- **Bear (P10-P25):** unavailable — Synthesis failed.

## Action recommendation
No action recommended. The Synthesis step failed; do not act on a partial brief.

## Watchlist
- Re-run drill-in once Synthesis is fixed (synthesis)

## Evidence
- (no evidence section — Synthesis failed)
"""


def _fallback_report(state: FinaqState) -> str:
    return _FALLBACK_TEMPLATE.format(
        ticker=state.get("ticker", "?"),
        thesis_name=(state.get("thesis") or {}).get("name", "unknown"),
        today=date.today().isoformat(),
    )


# --- Graph node --------------------------------------------------------------


async def run(state: FinaqState) -> dict:
    started_at = time.perf_counter()
    ticker = state.get("ticker", "")
    errors: list[str] = []

    # If literally all upstream agents failed, the synthesis is meaningless —
    # short-circuit to the fallback so we don't waste a synthesis-LLM call.
    has_any_input = any(
        bool(state.get(n)) for n in ("fundamentals", "filings", "news", "risk", "monte_carlo")
    )
    if not has_any_input:
        logger.warning(f"[synthesis] no upstream outputs for {ticker}; emitting fallback report")
        out = SynthesisOutput(
            report=_fallback_report(state),
            confidence="low",
            gaps=["all upstream agents produced no output"],
            watchlist=["re-run drill-in once upstream agents recover (synthesis)"],
            errors=["no upstream outputs"],
        )
    else:
        try:
            llm_raw = await asyncio.to_thread(_call_llm, state)
            normalised = _coerce_to_synthesis_output(llm_raw)
            out = SynthesisOutput.model_validate({**normalised, "errors": errors})
        except Exception as e:
            logger.error(f"[synthesis] LLM call failed for {ticker}: {e}")
            errors.append(f"llm: {e}")
            out = SynthesisOutput(
                report=_fallback_report(state),
                confidence="low",
                gaps=[],
                watchlist=[],
                errors=errors,
            )

    return {
        "report": out.report,
        "synthesis_confidence": out.confidence,
        "gaps": out.gaps,
        "watchlist": out.watchlist,
        "messages": [
            {
                "node": NODE,
                "event": "completed",
                "started_at": started_at,
                "completed_at": time.perf_counter(),
            }
        ],
    }


# --- CLI ---------------------------------------------------------------------


async def _cli(ticker: str, thesis_slug: str = "ai_cake") -> None:
    """Standalone CLI: runs Fundamentals + Filings + News + Risk + MC end to end,
    then Synthesis. Slow (~2 min) but the only way to test on real state."""
    from agents import build_graph

    thesis = json.loads(Path(f"theses/{thesis_slug}.json").read_text())
    graph = build_graph()
    final = await graph.ainvoke({"ticker": ticker, "thesis": thesis})
    print(final.get("report", "(no report)"))
    print("\n=== GAPS ===")
    print(final.get("gaps", []))
    print("\n=== WATCHLIST ===")
    print(final.get("watchlist", []))
    print("\n=== CONFIDENCE ===")
    print(final.get("synthesis_confidence", "?"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agents.synthesis TICKER [thesis_slug]", file=sys.stderr)
        sys.exit(1)
    ticker = sys.argv[1].upper()
    thesis_slug = sys.argv[2] if len(sys.argv) > 2 else "ai_cake"
    asyncio.run(_cli(ticker, thesis_slug))
