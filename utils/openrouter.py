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
   for the accumulator contract. Since schema v6 the interceptor also
   writes one `llm_calls` row per call (model, latency, tokens, cost,
   truncated prompt/response) so failed runs stay debuggable even when
   LangSmith tracing was off at run time. Since schema v9 that row is
   written in two phases — inserted as `running` with the FULL system +
   user messages before the SDK call, closed with tokens, cost, latency
   and the assistant message afterwards — so the live Mission Control
   view can show the prompt entering a model while it is still answering.
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
    import json
    import time

    from data import state as state_db
    from data.state import node_telemetry_var
    from utils.models import compute_cost

    # `client.chat.completions` is an attribute namespace, not a method —
    # we patch its `create` to capture usage on the way out.
    original_create = client.chat.completions.create

    def _prompt_excerpt(messages: Any) -> str:
        """Serialize only up to the storage cap — synthesis prompts run to
        hundreds of KB while record_llm_call clips to LLM_EXCERPT_MAX_CHARS,
        so dumping the full list per call would be pure waste."""
        parts: list[str] = []
        total = 0
        for m in messages or []:
            s = json.dumps(m, default=str)
            parts.append(s)
            total += len(s)
            if total >= state_db.LLM_EXCERPT_MAX_CHARS:
                break
        return "[" + ", ".join(parts) + "]"

    def _node_name(accumulator: Any) -> str:
        return str(accumulator.get("node") or "") if isinstance(accumulator, dict) else ""

    def _begin_trace_row(accumulator: Any, kwargs: dict) -> int | None:
        """Schema v9 — phase 1 of the per-call trace row, written BEFORE the
        SDK call: status `running`, the full system + user messages (stored
        through prompt_blobs) and the clipped excerpt for the pre-v9 readers.
        Best-effort: None when the write failed, so phase 2 falls back to the
        one-shot `record_llm_call`. run_id is None for calls outside any run."""
        try:
            return state_db.start_llm_call(
                run_id=state_db.current_run_id.get(None),
                node=_node_name(accumulator),
                model=kwargs.get("model") or "",
                messages=kwargs.get("messages"),
                prompt_excerpt=_prompt_excerpt(kwargs.get("messages")),
            )
        except Exception:
            return None

    def _end_trace_row(
        call_id: int | None,
        accumulator: Any,
        kwargs: dict,
        latency_s: float,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        response_excerpt: str,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        """Phase 2 — close the row with tokens, cost, latency, the assistant
        message and status completed | failed. A lost trace row never breaks
        the call; when phase 1 failed we still land a one-shot row (schema v6
        shape) so the run stays debuggable. When phase 1 succeeded and THIS
        update fails, the row stays `running` for the stale sweeper to close
        — deliberately no one-shot fallback, which would duplicate the call."""
        import contextlib

        with contextlib.suppress(Exception):
            if call_id is not None:
                state_db.finish_llm_call(
                    call_id,
                    status=status,
                    latency_s=latency_s,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost,
                    response_excerpt=response_excerpt,
                    response_text=response_excerpt if status == "completed" else None,
                    error=error,
                )
                return
            state_db.record_llm_call(
                run_id=state_db.current_run_id.get(None),
                node=_node_name(accumulator),
                model=kwargs.get("model") or "",
                latency_s=latency_s,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                prompt_excerpt=_prompt_excerpt(kwargs.get("messages")),
                response_excerpt=response_excerpt,
            )

    def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
        accumulator = node_telemetry_var.get(None)
        # Phase 1 (schema v9): the in-flight call is visible to the live
        # view before the model answers. Skipped outside a node, same
        # gating as the accumulator, so ad-hoc SDK calls write no ghost rows.
        call_id = _begin_trace_row(accumulator, kwargs) if accumulator is not None else None
        t0 = time.perf_counter()
        try:
            resp = original_create(*args, **kwargs)
        except Exception as call_exc:
            # A run that failed BECAUSE of LLM errors is exactly when the
            # trace matters — record the failed attempt, then re-raise.
            if accumulator is not None:
                _end_trace_row(
                    call_id, accumulator, kwargs, time.perf_counter() - t0,
                    0, 0, 0.0, response_excerpt=f"CALL FAILED: {call_exc}",
                    status="failed", error=str(call_exc),
                )
            raise
        latency_s = time.perf_counter() - t0
        if accumulator is None:
            return resp  # not inside a node — nothing to record
        tokens_in = tokens_out = 0
        cost = 0.0
        try:
            usage = getattr(resp, "usage", None) or {}
            tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
            cost = compute_cost(kwargs.get("model") or "", tokens_in, tokens_out)
            accumulator["tokens_in"] += tokens_in
            accumulator["tokens_out"] += tokens_out
            accumulator["cost_usd"] += cost
            accumulator["n_calls"] += 1
        except Exception:
            # Telemetry must never break the actual LLM call. A failed
            # accumulator is a debugging miss, not an outage.
            pass
        response_excerpt = ""
        try:
            choices = getattr(resp, "choices", None) or []
            if choices:
                response_excerpt = str(
                    getattr(getattr(choices[0], "message", None), "content", "") or ""
                )
        except Exception:
            pass
        _end_trace_row(
            call_id, accumulator, kwargs, latency_s, tokens_in, tokens_out, cost,
            response_excerpt, status="completed",
        )
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
