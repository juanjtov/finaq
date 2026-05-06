"""OpenRouter client factory. All FINAQ LLM calls flow through this.

Two responsibilities:
1. **LangSmith tracing** — when `LANGSMITH_TRACING=true`, the returned
   client is wrapped with `langsmith.wrappers.wrap_openai` so each
   `chat.completions.create(...)` call records as a child run under
   the active LangGraph node, with full prompt + response + token
   counts visible in the LangSmith UI.
2. **Per-node telemetry** — a thin interceptor reads `response.usage`
   and accumulates tokens + cost into a `ContextVar` set by
   `agents._safe_node`. Totals are written to `data_cache/state.db`
   on node exit so the Run Inspector page can show per-node cost
   without re-querying LangSmith. See `data.state.node_telemetry_var`
   for the accumulator contract.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MAX_RETRIES = 3


def _maybe_wrap_for_langsmith(client: OpenAI) -> Any:
    """When `LANGSMITH_TRACING=true` is set, wrap the client so individual
    `chat.completions.create(...)` calls show up as child runs in the
    LangSmith UI (with prompt + response + token counts), nested under
    the active LangGraph node.

    Without this wrap, LangSmith only sees the LangGraph node boundaries
    — every per-LLM "Input/Output" panel reads "Unknown / No message
    content" because the SDK calls underneath are invisible. We use the
    raw OpenAI SDK rather than LangChain's ChatOpenAI; `wrap_openai`
    bridges that gap.
    """
    if not os.getenv("LANGSMITH_TRACING"):
        return client
    try:
        from langsmith.wrappers import wrap_openai
    except ImportError:
        return client
    return wrap_openai(client)


def _install_telemetry_interceptor(client: Any) -> Any:
    """Patch `client.chat.completions.create` to read `response.usage`
    and accumulate tokens + cost into the `ContextVar` set by
    `agents._safe_node`.

    Per-node telemetry pattern:
      - `_safe_node` enters a context that initialises a fresh
        accumulator dict and binds it to a ContextVar
      - Every LLM call inside the node (one or many) adds its token
        counts + cost to that accumulator
      - On node exit, `_safe_node` reads the totals and writes them to
        `node_runs.tokens_in / tokens_out / cost_usd`

    The interceptor is a no-op when no node is active (e.g. ad-hoc
    direct calls from `/analyze` or `agents.qa.ask` outside the graph),
    so direct calls don't crash the SDK.
    """
    from data.state import node_telemetry_var
    from utils.models import compute_cost

    # `client.chat.completions` is an attribute namespace, not a method —
    # we patch its `create` to capture usage on the way out.
    original_create = client.chat.completions.create

    def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
        resp = original_create(*args, **kwargs)
        accumulator = node_telemetry_var.get(None)
        if accumulator is None:
            return resp  # not inside a node — nothing to record
        try:
            usage = getattr(resp, "usage", None) or {}
            tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
            model = kwargs.get("model") or ""
            cost = compute_cost(model, tokens_in, tokens_out)
            accumulator["tokens_in"] += tokens_in
            accumulator["tokens_out"] += tokens_out
            accumulator["cost_usd"] += cost
            accumulator["n_calls"] += 1
        except Exception:
            # Telemetry must never break the actual LLM call. A failed
            # accumulator is a debugging miss, not an outage.
            pass
        return resp

    client.chat.completions.create = _wrapped_create
    return client


def get_client() -> Any:
    """Return an OpenAI-compatible client pointed at OpenRouter, optionally
    wrapped for LangSmith tracing + per-node telemetry.

    The OpenAI SDK has built-in retries; we set `max_retries=3` here so
    callers don't need to wrap LLM calls with `tenacity_retry` separately.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    raw = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=OPENROUTER_MAX_RETRIES,
    )
    traced = _maybe_wrap_for_langsmith(raw)
    return _install_telemetry_interceptor(traced)
