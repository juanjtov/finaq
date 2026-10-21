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

import sqlite3
from types import SimpleNamespace

import pytest

from data import state as state_db
from utils.models import MODEL_PRICING
from utils.openrouter import _install_telemetry_interceptor

# Any model with a row in the price table — picked at runtime so the tests
# never hard-code a model id.
PRICED_MODEL = next(iter(MODEL_PRICING))


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


# --- Schema v6 — per-call llm_calls trace rows -----------------------------


def test_interceptor_writes_llm_call_row_with_run_id(tmp_path, monkeypatch):
    """Inside a graph node, every chat-completion call must land one
    llm_calls row attributed to the active run + node, with model,
    latency, tokens, cost, and truncated prompt/response excerpts."""
    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "trace.db")
    run_id = state_db.start_graph_run("NVDA", "ai_cake")
    run_token = state_db.current_run_id.set(run_id)
    accumulator = state_db.new_node_telemetry(node="fundamentals")
    node_token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(1000, 500))
        )
        client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": "summarise the KPIs"}],
        )
    finally:
        state_db.node_telemetry_var.reset(node_token)
        state_db.current_run_id.reset(run_token)

    calls = state_db.llm_calls_for_run(run_id)
    assert len(calls) == 1
    c = calls[0]
    assert c["node"] == "fundamentals"
    assert c["model"] == "anthropic/claude-haiku-4.5"
    assert c["tokens_in"] == 1000
    assert c["tokens_out"] == 500
    assert c["cost_usd"] > 0
    assert c["latency_s"] >= 0.0
    assert "summarise the KPIs" in c["prompt_excerpt"]
    assert c["response_excerpt"] == "ok"


def test_interceptor_no_llm_call_row_without_accumulator(tmp_path, monkeypatch):
    """No node active → no trace row (same gating as the accumulator, so
    ad-hoc SDK calls never write ghost rows)."""
    import sqlite3

    db = tmp_path / "trace.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    assert state_db.node_telemetry_var.get() is None
    client = _install_telemetry_interceptor(_fake_client(_fake_response(10, 5)))
    client.chat.completions.create(model="anthropic/claude-haiku-4.5", messages=[])
    if db.exists():
        with sqlite3.connect(db) as conn:
            n = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
        assert n == 0


def test_interceptor_records_null_run_id_outside_graph(tmp_path, monkeypatch):
    """The CIO planner binds an accumulator without a graph run — the
    trace row is still written, with run_id NULL."""
    import sqlite3

    db = tmp_path / "trace.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    accumulator = state_db.new_node_telemetry()
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(100, 50))
        )
        client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=[]
        )
    finally:
        state_db.node_telemetry_var.reset(token)
    with sqlite3.connect(db) as conn:
        rows = list(conn.execute("SELECT run_id, node FROM llm_calls"))
    assert len(rows) == 1
    assert rows[0][0] is None
    assert rows[0][1] == ""


def test_interceptor_trace_write_failure_does_not_break_call(monkeypatch):
    """Same invariant as the accumulator: a failed llm_calls write is a
    debugging miss, never an outage."""
    def _boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(state_db, "record_llm_call", _boom)
    accumulator = state_db.new_node_telemetry(node="news")
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(100, 50))
        )
        resp = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=[]
        )
        assert resp.usage.prompt_tokens == 100
        # The accumulator still recorded the call even though the trace
        # row was lost.
        assert accumulator["n_calls"] == 1
    finally:
        state_db.node_telemetry_var.reset(token)


def test_interceptor_records_failed_call_and_reraises(tmp_path, monkeypatch):
    """A call that raises (429/500 after SDK retries) must still land an
    llm_calls row marked CALL FAILED — a run that died BECAUSE of LLM
    errors is exactly when the trace matters — and the exception must
    propagate unchanged."""
    import pytest as _pytest

    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "trace.db")
    run_id = state_db.start_graph_run("NVDA", "ai_cake")
    run_token = state_db.current_run_id.set(run_id)
    accumulator = state_db.new_node_telemetry(node="synthesis")
    node_token = state_db.node_telemetry_var.set(accumulator)

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("upstream 500")

    from types import SimpleNamespace as _NS

    client = _install_telemetry_interceptor(
        _NS(chat=_NS(completions=_Completions()))
    )
    try:
        with _pytest.raises(RuntimeError, match="upstream 500"):
            client.chat.completions.create(
                model="anthropic/claude-haiku-4.5",
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        state_db.node_telemetry_var.reset(node_token)
        state_db.current_run_id.reset(run_token)

    calls = state_db.llm_calls_for_run(run_id)
    assert len(calls) == 1
    assert calls[0]["node"] == "synthesis"
    assert "CALL FAILED: upstream 500" in calls[0]["response_excerpt"]
    assert calls[0]["tokens_in"] == 0
    # The failed attempt does NOT count toward the accumulator (no usage).
    assert accumulator["n_calls"] == 0


def test_interceptor_prompt_excerpt_stops_at_cap(tmp_path, monkeypatch):
    """The prompt excerpt serializes messages only up to the storage cap —
    a synthesis-sized messages list must not be fully json.dumps'd."""
    monkeypatch.setattr(state_db, "DB_PATH", tmp_path / "trace.db")
    accumulator = state_db.new_node_telemetry(node="filings")
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _install_telemetry_interceptor(
            _fake_client(_fake_response(10, 5))
        )
        big = [{"role": "user", "content": "x" * 5000} for _ in range(50)]
        client.chat.completions.create(
            model="anthropic/claude-haiku-4.5", messages=big
        )
    finally:
        state_db.node_telemetry_var.reset(token)
    import sqlite3 as _sq

    with _sq.connect(tmp_path / "trace.db") as conn:
        excerpt = conn.execute(
            "SELECT prompt_excerpt FROM llm_calls"
        ).fetchone()[0]
    assert len(excerpt) == state_db.LLM_EXCERPT_MAX_CHARS
    assert excerpt.startswith('[{"role"')


# --- Schema v9 — two-phase llm_calls rows with full messages ----------------


def _rows(db, sql: str, *args) -> list[dict]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, args)]


def _client_with(create_fn):
    return _install_telemetry_interceptor(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))
    )


def test_call_row_is_running_with_full_messages_before_the_sdk_returns(tmp_path, monkeypatch):
    """The live view's contract: while the model is still answering, the
    llm_calls row already exists as `running` with the FULL system and user
    text; afterwards it carries tokens, cost, latency and the assistant
    message — and the clipped excerpts the pre-v9 readers expect."""
    db = tmp_path / "trace.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    run_id = state_db.start_graph_run("NVDA", "ai_cake")
    system = "You are the fundamentals analyst. " * 200  # ~7 KB: past the 2,000-char excerpt
    user = "Summarise the KPIs for NVDA. " * 100
    seen: dict = {}

    def _blocking_create(**kwargs):
        # The SDK call is in flight — this is what a 1 s poll would see.
        rows = _rows(db, "SELECT * FROM llm_calls")
        seen["rows"] = rows
        seen["messages"] = state_db.llm_messages_for_call(rows[0]["id"]) if rows else []
        return _fake_response(1000, 500)

    client = _client_with(_blocking_create)
    run_token = state_db.current_run_id.set(run_id)
    node_token = state_db.node_telemetry_var.set(state_db.new_node_telemetry(node="fundamentals"))
    try:
        client.chat.completions.create(
            model=PRICED_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
    finally:
        state_db.node_telemetry_var.reset(node_token)
        state_db.current_run_id.reset(run_token)

    [mid] = seen["rows"]
    assert (mid["status"], mid["ended_at"], mid["node"], mid["run_id"]) == (
        "running",
        None,
        "fundamentals",
        run_id,
    )
    assert mid["model"] == PRICED_MODEL and mid["attempt"] == 1
    assert [(m["role"], m["text"]) for m in seen["messages"]] == [
        ("system", system),
        ("user", user),
    ]

    [after] = state_db.llm_calls_for_run(run_id)
    assert after["id"] == mid["id"]  # closed in place, not re-inserted
    assert after["status"] == "completed" and after["ended_at"] and after["error"] is None
    assert (after["tokens_in"], after["tokens_out"]) == (1000, 500)
    assert after["cost_usd"] > 0 and after["latency_s"] >= 0.0
    messages = state_db.llm_messages_for_call(after["id"])
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[-1]["text"] == "ok"
    # Old readers (Run Inspector, Mission Control) still get their excerpts.
    assert after["prompt_excerpt"].startswith('[{"role"')
    assert len(after["prompt_excerpt"]) <= state_db.LLM_EXCERPT_MAX_CHARS
    assert user not in after["prompt_excerpt"]  # the excerpt clipped it; the blob has it
    assert after["response_excerpt"] == "ok"


def test_system_blob_is_stored_once_across_two_calls(tmp_path, monkeypatch):
    db = tmp_path / "trace.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    system = {"role": "system", "content": "persona " * 500}
    client = _client_with(lambda **kw: _fake_response(10, 5))
    token = state_db.node_telemetry_var.set(state_db.new_node_telemetry(node="news"))
    try:
        for user in ("first question", "second question"):
            client.chat.completions.create(
                model=PRICED_MODEL, messages=[system, {"role": "user", "content": user}]
            )
    finally:
        state_db.node_telemetry_var.reset(token)
    blobs = _rows(db, "SELECT content, seen_count FROM prompt_blobs ORDER BY first_seen, content")
    by_content = {b["content"]: b["seen_count"] for b in blobs}
    # system ×2, two user prompts, and the identical "ok" response ×2.
    assert by_content == {system["content"]: 2, "first question": 1, "second question": 1, "ok": 2}
    assert _rows(db, "SELECT COUNT(*) AS n FROM llm_messages")[0]["n"] == 6


def test_failed_call_ends_failed_with_error_and_reraises(tmp_path, monkeypatch):
    db = tmp_path / "trace.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)
    run_id = state_db.start_graph_run("NVDA", "ai_cake")

    def _explode(**kwargs):
        raise RuntimeError("upstream 500")

    client = _client_with(_explode)
    run_token = state_db.current_run_id.set(run_id)
    node_token = state_db.node_telemetry_var.set(state_db.new_node_telemetry(node="synthesis"))
    try:
        with pytest.raises(RuntimeError, match="upstream 500"):
            client.chat.completions.create(
                model=PRICED_MODEL,
                messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            )
    finally:
        state_db.node_telemetry_var.reset(node_token)
        state_db.current_run_id.reset(run_token)

    [row] = state_db.llm_calls_for_run(run_id)
    assert (row["status"], row["error"]) == ("failed", "upstream 500")
    assert row["ended_at"] is not None and row["tokens_in"] == 0
    assert row["response_excerpt"] == "CALL FAILED: upstream 500"  # pre-v9 readers' marker
    # The request messages were captured before the failure; no assistant row.
    assert [m["role"] for m in state_db.llm_messages_for_call(row["id"])] == ["system", "user"]


def test_entry_write_failure_falls_back_to_a_one_shot_row(tmp_path, monkeypatch):
    """Phase 1 failing (disk full, locked DB) must not lose the call: phase 2
    lands the schema-v6-shaped one-shot row instead."""
    db = tmp_path / "trace.db"
    monkeypatch.setattr(state_db, "DB_PATH", db)

    def _boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(state_db, "start_llm_call", _boom)
    client = _client_with(lambda **kw: _fake_response(100, 50))
    token = state_db.node_telemetry_var.set(state_db.new_node_telemetry(node="risk"))
    try:
        resp = client.chat.completions.create(
            model=PRICED_MODEL, messages=[{"role": "user", "content": "hi"}]
        )
    finally:
        state_db.node_telemetry_var.reset(token)
    assert resp.usage.prompt_tokens == 100
    [row] = _rows(db, "SELECT * FROM llm_calls")
    assert (row["node"], row["status"], row["tokens_in"]) == ("risk", "completed", 100)
    assert "hi" in row["prompt_excerpt"] and row["response_excerpt"] == "ok"
    assert state_db.llm_messages_for_call(row["id"]) == []  # no messages on the fallback path


def test_both_trace_writes_failing_never_breaks_the_call(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(state_db, "start_llm_call", _boom)
    monkeypatch.setattr(state_db, "record_llm_call", _boom)
    monkeypatch.setattr(state_db, "finish_llm_call", _boom)
    accumulator = state_db.new_node_telemetry(node="news")
    token = state_db.node_telemetry_var.set(accumulator)
    try:
        client = _client_with(lambda **kw: _fake_response(100, 50))
        resp = client.chat.completions.create(model=PRICED_MODEL, messages=[])
    finally:
        state_db.node_telemetry_var.reset(token)
    assert resp.usage.prompt_tokens == 100
    assert accumulator["n_calls"] == 1
