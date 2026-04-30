"""Per-agent free-text Q&A — the `ask()` path.

Each upstream agent has a `run()` method that produces structured output
during a drill-in. This module adds a parallel **`ask()`** path: a single
cheap LLM call that answers a free-text question grounded in that agent's
existing structured output (or, for Filings, a fresh RAG retrieval scoped
to the user's question).

Used by:
  - The dashboard's Direct Agent panel (Step 8e)
  - Phase 1 Telegram per-agent slash commands (`/fundamentals NVDA "..."`)
  - Future Triage / Discovery agents that want focused answers without
    re-running the full graph

Why a separate module rather than a method on each agent file:
  - The 4 functions share boilerplate (prompt loading, JSON parsing,
    citation coercion, error fallback) — DRY rule of three is satisfied.
  - The agent files stay focused on the `run()` graph-node contract.
  - One model env var (`MODEL_AGENT_QA`, defaulting to Haiku) governs the
    cost of the entire Q&A surface.

Public API: `await ask(state, agent, question)` returns an AgentAnswer.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from utils import logger
from utils.models import MODEL_AGENT_QA
from utils.openrouter import get_client
from utils.schemas import AgentAnswer, Evidence
from utils.state import FinaqState

PROMPTS_DIR = Path(__file__).parent / "prompts"
AGENT_NAMES = ("fundamentals", "filings", "news", "risk")
AgentName = Literal["fundamentals", "filings", "news", "risk"]
# Bumped from 800 → 1500 in late Step 8: with 800 the LLM occasionally hit
# the limit mid-JSON and produced an "Unterminated string" parse error.
# 1500 gives enough headroom for a ~800-word answer plus 4-5 citations.
LLM_MAX_TOKENS = 1500

# Filings ask() retrieves more chunks than the run() default — the question is
# more specific so we cast a wider net to find the supporting passage.
FILINGS_ASK_K = 8
FILINGS_ASK_CANDIDATE_POOL = 60


# --- Prompt loading (cached at module import) -------------------------------

_SYSTEM_PROMPTS: dict[str, str] = {
    name: (PROMPTS_DIR / f"qa_{name}.md").read_text() for name in AGENT_NAMES
}


# --- Helpers ----------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text


def _coerce_answer(
    agent: AgentName, question: str, raw_json: dict, errors: list[str]
) -> AgentAnswer:
    """Turn the LLM's raw JSON into an AgentAnswer. Be liberal in what we
    accept — strings missing `excerpt` / unknown keys / extra fluff get
    dropped rather than crash the call."""
    answer = str(raw_json.get("answer", "")).strip()
    if not answer:
        answer = "(empty answer — LLM did not produce a response)"
        errors = errors + ["empty answer"]

    # If `_parse_recovery` was injected by the JSON-recovery path, surface
    # that to the user as an error so they know they're seeing a partial
    # answer (citations were dropped due to truncated JSON).
    if raw_json.get("_parse_recovery"):
        errors = errors + [f"parse-recovery: {raw_json['_parse_recovery']}"]

    citations: list[Evidence] = []
    for c in raw_json.get("citations") or []:
        if not isinstance(c, dict):
            continue
        try:
            ev = Evidence(
                source=str(c.get("source", "unknown")),
                accession=c.get("accession"),
                item=c.get("item"),
                url=c.get("url"),
                excerpt=c.get("excerpt"),
                note=c.get("note"),
                as_of=c.get("as_of"),
            )
            citations.append(ev)
        except Exception as e:  # tolerate per-citation errors
            logger.warning(f"[qa.{agent}] dropping malformed citation: {c} ({e})")

    return AgentAnswer(
        agent=agent,
        question=question,
        answer=answer,
        citations=citations,
        errors=errors,
    )


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


# --- Per-agent context builders --------------------------------------------


def _fundamentals_context(state: FinaqState) -> str:
    fund = state.get("fundamentals") or {}
    return (
        f"FUNDAMENTALS payload:\n"
        f"summary: {_truncate(fund.get('summary', ''), 600)}\n"
        f"kpis: {json.dumps(fund.get('kpis') or {}, indent=2, default=str)[:1500]}\n"
        f"projections: {json.dumps(fund.get('projections') or {}, default=str)[:1000]}\n"
    )


def _news_context(state: FinaqState) -> str:
    news = state.get("news") or {}
    parts = [f"NEWS payload:\nsummary: {_truncate(news.get('summary', ''), 600)}"]
    cats = news.get("catalysts") or []
    concerns = news.get("concerns") or []
    if cats:
        parts.append(f"\ncatalysts ({len(cats)}):")
        for c in cats[:8]:
            parts.append(
                f"  - [{c.get('sentiment', '?')} {c.get('as_of', '?')}] "
                f"{c.get('title', '')} ({c.get('url', '')})"
            )
    if concerns:
        parts.append(f"\nconcerns ({len(concerns)}):")
        for c in concerns[:8]:
            parts.append(
                f"  - [{c.get('sentiment', '?')} {c.get('as_of', '?')}] "
                f"{c.get('title', '')} ({c.get('url', '')})"
            )
    return "\n".join(parts)


def _risk_context(state: FinaqState) -> str:
    risk = state.get("risk") or {}
    parts = [
        f"RISK payload:\n"
        f"level: {risk.get('level', '?')} (score {risk.get('score_0_to_10', '?')})",
        f"summary: {_truncate(risk.get('summary', ''), 500)}",
    ]
    top = risk.get("top_risks") or []
    if top:
        parts.append(f"top_risks ({len(top)}):")
        for t in top:
            parts.append(
                f"  - [sev={t.get('severity', '?')} sources={t.get('sources', [])}] "
                f"{t.get('title', '')} — {_truncate(t.get('explanation', ''), 200)}"
            )
    cs = risk.get("convergent_signals") or []
    if cs:
        parts.append(f"convergent_signals ({len(cs)}):")
        for s in cs:
            parts.append(
                f"  - [{s.get('sources', [])}] {s.get('theme', '')} — "
                f"{_truncate(s.get('explanation', ''), 200)}"
            )
    breaches = risk.get("threshold_breaches") or []
    if breaches:
        parts.append(f"threshold_breaches ({len(breaches)}):")
        for b in breaches:
            parts.append(
                f"  - {b.get('signal', '')} {b.get('operator', '')} "
                f"{b.get('threshold_value', '')} (observed={b.get('observed_value', '')}, "
                f"source={b.get('source', '')})"
            )
    return "\n".join(parts)


def _filings_context_from_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "RETRIEVED FILING CHUNKS: (none — RAG returned 0 hits)"
    parts = ["RETRIEVED FILING CHUNKS BY RELEVANCE:"]
    for i, ch in enumerate(chunks, start=1):
        md = ch.get("metadata") or {}
        parts.append(
            f"\n[chunk {i}] accession={md.get('accession', '?')} "
            f"item={md.get('item_label', '?')!r} filed_date={md.get('filed_date', '?')}\n"
            f"{ch.get('text', '')}"
        )
    return "\n".join(parts)


def _build_user_prompt(
    ticker: str, thesis: dict, question: str, agent_context: str
) -> str:
    parts = [
        f"TICKER: {ticker}",
        f"ACTIVE THESIS: {thesis.get('name', 'unknown')}",
        f"THESIS SUMMARY: {_truncate(thesis.get('summary', ''), 400)}",
        "",
        f"QUESTION: {question}",
        "",
        agent_context,
        "",
        "ANSWER NOW. STRICT JSON ONLY.",
    ]
    return "\n".join(parts)


# --- LLM call ---------------------------------------------------------------


_ANSWER_FIELD_RE = re.compile(r'"answer"\s*:\s*"(.+?)"\s*[,}]', re.DOTALL)


def _parse_llm_response(raw: str) -> dict:
    """Parse the LLM's JSON output with a graceful fallback.

    Phase 0 observed failure mode: the LLM hits max_tokens mid-JSON and
    produces an unterminated string (`json.JSONDecodeError: Unterminated
    string starting at...`). Strict json.loads then crashes the whole call.

    Recovery strategy (in order):
      1. Strip code fences and try `json.loads`.
      2. On JSONDecodeError, regex-extract the `"answer": "..."` field —
         most useful information for the user is the prose answer; the
         citations array is what truncated.
      3. On total failure, return the raw text as the answer so the user
         sees something rather than a stack-trace error message.
    """
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
        return parsed
    except (json.JSONDecodeError, ValueError) as parse_err:
        logger.warning(
            f"[qa] strict JSON parse failed ({parse_err}); attempting graceful recovery"
        )
        # Try regex-extracting the answer field
        m = _ANSWER_FIELD_RE.search(cleaned)
        if m:
            recovered = m.group(1).replace('\\"', '"').replace("\\n", "\n")
            return {
                "answer": recovered,
                "citations": [],
                "_parse_recovery": "regex-extracted answer; citations dropped due to JSON truncation",
            }
        # Last resort — show the raw text trimmed
        return {
            "answer": cleaned[:1500] if cleaned else "(empty LLM response)",
            "citations": [],
            "_parse_recovery": f"unparseable JSON: {parse_err}",
        }


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_AGENT_QA,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=LLM_MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_llm_response(raw)


# --- Public API -------------------------------------------------------------


async def ask(state: FinaqState, agent: AgentName, question: str) -> AgentAnswer:
    """Answer `question` over the upstream output of `agent`.

    Per-agent semantics:
      - fundamentals / risk / news: pure synthesis over `state.<agent>`. No
        external calls. If the agent's payload is empty (run never produced
        output), returns a graceful "no data" answer.
      - filings: re-runs the RAG retrieval with `question` as the query
        (using the same hybrid retriever as the drill-in). The retrieved
        chunks are scoped to `state.ticker`.
    """
    if agent not in AGENT_NAMES:
        raise ValueError(f"unknown agent {agent!r}; expected one of {AGENT_NAMES}")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    ticker = state.get("ticker", "")
    thesis = state.get("thesis") or {}
    errors: list[str] = []

    # Build the agent-specific context block
    if agent == "fundamentals":
        if not state.get("fundamentals"):
            return AgentAnswer(
                agent=agent,
                question=question,
                answer="No Fundamentals data in this drill-in's state. Run a drill-in first.",
                errors=["state.fundamentals is empty"],
            )
        agent_context = _fundamentals_context(state)
    elif agent == "news":
        if not state.get("news"):
            return AgentAnswer(
                agent=agent,
                question=question,
                answer="No News data in this drill-in's state. Run a drill-in first.",
                errors=["state.news is empty"],
            )
        agent_context = _news_context(state)
    elif agent == "risk":
        if not state.get("risk"):
            return AgentAnswer(
                agent=agent,
                question=question,
                answer="No Risk synthesis in this drill-in's state. Run a drill-in first.",
                errors=["state.risk is empty"],
            )
        agent_context = _risk_context(state)
    elif agent == "filings":
        # Re-run RAG with the user's question; cheaper than re-running the full
        # Filings agent because we use Haiku and skip the synthesis.
        from data.chroma import query as chroma_query

        if not ticker:
            return AgentAnswer(
                agent=agent,
                question=question,
                answer="No ticker set — cannot run Filings RAG retrieval.",
                errors=["state.ticker is empty"],
            )
        try:
            chunks = await asyncio.to_thread(
                chroma_query,
                ticker,
                question,
                FILINGS_ASK_K,
                None,  # item_filter — no constraint here, let RAG choose
                FILINGS_ASK_CANDIDATE_POOL,
            )
        except Exception as e:
            logger.error(f"[qa.filings] retrieval failed: {e}")
            errors.append(f"retrieval: {e}")
            chunks = []
        if not chunks:
            return AgentAnswer(
                agent=agent,
                question=question,
                answer=(
                    f"No filings chunks retrieved for {ticker}. "
                    "Has this ticker been ingested into ChromaDB?"
                ),
                errors=errors + ["no chunks retrieved"],
            )
        agent_context = _filings_context_from_chunks(chunks)
    else:  # pragma: no cover — guarded by AGENT_NAMES check above
        raise AssertionError(f"unreachable: {agent}")

    user_prompt = _build_user_prompt(ticker, thesis, question, agent_context)
    system_prompt = _SYSTEM_PROMPTS[agent]

    try:
        raw = await asyncio.to_thread(_call_llm, system_prompt, user_prompt)
    except Exception as e:
        logger.error(f"[qa.{agent}] LLM call failed: {e}")
        errors.append(f"llm: {e}")
        return AgentAnswer(
            agent=agent,
            question=question,
            answer=f"LLM call failed: {e}",
            errors=errors,
        )

    return _coerce_answer(agent, question, raw, errors)
