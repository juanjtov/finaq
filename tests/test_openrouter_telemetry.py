"""Tier 1 unit tests for the per-node telemetry interceptor in
`utils/openrouter.py` (Step 10c.8).

The interceptor:
- Reads `response.usage.{prompt_tokens, completion_tokens}` after every
  `chat.completions.create(...)` call.
- Looks up the model's $/1M rate via `utils.models.compute_cost`.
- Adds tokens + cost to the ContextVar accumulator (`data.state.node_telemetry_var`)
  IF one is bound (i.e. we're inside a `_safe_node` invocation).
- Is a no-op when no node is active (direct calls outside the graph).

Tests use a hand-rolled fake OpenAI client whose `chat.completions.create`
returns a response with a `usage` attribute — same shape the real SDK
returns.
"""

from __future__ import annotations

from types import SimpleNamespace

from data import state as state_db
from utils.openrouter import _install_telemetry_interceptor


def _fake_response(prompt_tokens: int, completion_tokens: int):
    """Mimics the OpenAI SDK's response object shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _fake_client(response):
    """Build a minimal client whose `chat.completions.create` returns `response`."""
    class _Completions:
        def __init__(self):
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return response

    completions = _Completions()
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat, _completions=completions)


def test_interceptor_no_op_when_no_node_active():
    """Direct ad-hoc calls (e.g. outside _safe_node) must NOT crash and
    must NOT accumulate ghost telemetry."""
    client = _install_telemetry_interceptor(
        _fake_client(_fake_response(100, 50))
    )
    # ContextVar default is None — no node active.
    assert state_db.node_telemetry_var.get() is None
    resp = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5", messages=[]
    )
    # Returns the response unchanged (no exception).
    assert resp.usage.prompt_tokens == 100


def test_interceptor_accumulates_when_node_active():
    """Inside a node, the interceptor must add tokens + cost to the
    ContextVar accumulator. Each call increments n_calls."""
    accumulator = state_db.new_node_telemetry()
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(1000, 500))
        )
        client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=[]
        )
        # haiku-4.5 pricing: $1.00/1M input, $5.00/1M output → 1000*1e-6*$1 + 500*1e-6*$5 = $0.0035
        assert accumulator["tokens_in"] == 1000
        assert accumulator["tokens_out"] == 500
        assert accumulator["cost_usd"] > 0
        assert accumulator["n_calls"] == 1

        # Second call accumulates.
        client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=[]
        )
        assert accumulator["tokens_in"] == 2000
        assert accumulator["n_calls"] == 2
    finally:
        state_db.node_telemetry_var.reset(token)


def test_interceptor_handles_unknown_model_gracefully():
    """A model not in `MODEL_PRICING` must add tokens but cost=0 — the
    pipeline doesn't break when the user swaps to a model whose row
    hasn't been added to the price table yet."""
    accumulator = state_db.new_node_telemetry()
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(100, 50))
        )
        client.chat.completions.create(model="not-in-pricing-table", messages=[])
        assert accumulator["tokens_in"] == 100
        assert accumulator["cost_usd"] == 0.0
        assert accumulator["n_calls"] == 1
    finally:
        state_db.node_telemetry_var.reset(token)


def test_interceptor_handles_missing_usage_field():
    """Some responses (errors, streaming) have no `usage` attribute. The
    interceptor must not crash; tokens/cost stay at 0."""
    accumulator = state_db.new_node_telemetry()
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        # Build a response WITHOUT the usage attribute.
        no_usage = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        client = _install_telemetry_interceptor(_fake_client(no_usage))
        client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=[]
        )
        assert accumulator["tokens_in"] == 0
        assert accumulator["cost_usd"] == 0.0
    finally:
        state_db.node_telemetry_var.reset(token)


def test_interceptor_failure_in_accumulator_does_not_break_call():
    """Telemetry must never break the actual LLM call. If the accumulator
    can't be updated for some reason (corrupted dict, etc.), the response
    still gets returned to the caller."""
    bad_accumulator: dict = {}  # missing keys → KeyError on +=
    token = state_db.node_telemetry_var.set(bad_accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(100, 50))
        )
        # Should not raise.
        resp = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=[]
        )
        assert resp.usage.prompt_tokens == 100
    finally:
        state_db.node_telemetry_var.reset(token)


def test_compute_cost_known_model():
    """Sanity-check the pricing table arithmetic so a future edit can't
    silently introduce a 10x bug."""
    from utils.models import compute_cost

    # 1000 prompt tokens × $1/1M + 500 completion × $5/1M = $0.001 + $0.0025 = $0.0035
    cost = compute_cost("anthropic/claude-haiku-4.5", 1000, 500)
    assert abs(cost - 0.0035) < 1e-9


def test_compute_cost_unknown_model_returns_zero():
    from utils.models import compute_cost

    assert compute_cost("not-in-pricing-table", 1_000_000, 1_000_000) == 0.0
