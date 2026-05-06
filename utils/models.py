"""Typed registry of OpenRouter model strings, sourced from .env.

Swap any model by editing the corresponding env var — no code change required.
See .env.example for the canonical list of variables and 2026-04-26 defaults.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required env var {name} is not set. See .env.example.")
    return val


MODEL_TRIAGE: str = _required("MODEL_TRIAGE")
MODEL_FUNDAMENTALS: str = _required("MODEL_FUNDAMENTALS")
MODEL_FILINGS: str = _required("MODEL_FILINGS")
MODEL_NEWS: str = _required("MODEL_NEWS")
MODEL_RISK: str = _required("MODEL_RISK")
MODEL_SYNTHESIS: str = _required("MODEL_SYNTHESIS")
MODEL_ROUTER: str = _required("MODEL_ROUTER")
MODEL_ADHOC_THESIS: str = _required("MODEL_ADHOC_THESIS")
MODEL_JUDGE: str = _required("MODEL_JUDGE")
MODEL_AGENT_QA: str = _required("MODEL_AGENT_QA")
MODEL_EMBEDDINGS: str = _required("MODEL_EMBEDDINGS")


# --- Pricing for telemetry / cost tracking --------------------------------
#
# Maps OpenRouter model id → (prompt $/1M tokens, completion $/1M tokens).
# Used by `utils.openrouter`'s telemetry interceptor to compute per-LLM-call
# cost on the fly so `data_cache/state.db.node_runs.cost_usd` is populated
# for the Run Inspector + the `/status` "today's spend" line.
#
# This table is documentation of *config defaults*, not a behaviour claim
# (similar to CLAUDE.md §3's routing table) — it's the only place outside
# `.env` where specific model ids appear, and it's purely a price lookup.
# Add new rows when you swap models in `.env`. Missing models cost-default
# to 0 so unknown models don't break the pipeline (you'll see `cost=$0.00`
# until the row is added).
#
# Rates sourced from OpenRouter's public pricing page; refresh quarterly.

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic Claude family (2026-04-26 OpenRouter rates, $/1M tokens)
    "anthropic/claude-opus-4.7":   (15.00, 75.00),
    "anthropic/claude-sonnet-4.6": (3.00,  15.00),
    "anthropic/claude-haiku-4.5":  (1.00,  5.00),
    "anthropic/claude-3.5-haiku":  (0.80,  4.00),
    # OpenAI
    "openai/gpt-5.4-mini":         (0.15,  0.60),
    "openai/gpt-4o-mini":          (0.15,  0.60),
    # Google
    "google/gemini-2.5-flash":     (0.075, 0.30),
    # Embeddings billed differently — nominal placeholder so dashboards
    # don't show $0 on every embedding-heavy run; refine if needed.
    "text-embedding-3-small":      (0.02,  0.0),
}


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost for a single LLM call given the OpenRouter model id
    and the prompt/completion token counts. Unknown models return 0.0 so
    the pipeline doesn't break when the user swaps to a model not yet in
    `MODEL_PRICING` — the cost just won't be tracked until the row is
    added (visible as `$0.00` in the Run Inspector / Mission Control)."""
    rates = MODEL_PRICING.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (tokens_in / 1_000_000.0) * in_rate + (tokens_out / 1_000_000.0) * out_rate
