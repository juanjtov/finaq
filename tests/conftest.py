"""Test fixtures and env-stub setup.

Loads the real .env first so integration tests see the user's keys + model
strings, then fills in non-empty placeholders for any MODEL_* var still
missing so unit tests work without a populated .env.

Also redirects the state.db telemetry layer to a tmp_path-scoped file for
the duration of each test session so unit tests don't pollute the user's
real `data_cache/state.db`.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Generic non-empty placeholder for any MODEL_* var not in .env.
_STUB = "test-stub-model"

_MODEL_STUB_VARS = (
    "MODEL_TRIAGE",
    "MODEL_FUNDAMENTALS",
    "MODEL_FILINGS",
    "MODEL_NEWS",
    "MODEL_RISK",
    "MODEL_SYNTHESIS",
    "MODEL_ROUTER",
    "MODEL_ADHOC_THESIS",
    "MODEL_JUDGE",
    "MODEL_AGENT_QA",
    "MODEL_EMBEDDINGS",
)

for _name in _MODEL_STUB_VARS:
    os.environ.setdefault(_name, _STUB)

# Never trace test runs to LangSmith. Stubbed graph runs would pollute the
# project and burn the free tier's monthly unique-trace quota (observed
# exhausted: every traced call then logs a 429 retry spew into test output).
# Set to "" (not pop) — the load_dotenv() calls in utils/ run later and would
# re-insert a popped var from .env, but never override an existing one.
# Tests that assert tracing-enabled behaviour monkeypatch the var themselves.
os.environ["LANGSMITH_TRACING"] = ""

# Same treatment for the freshness-gate kill-switch (an operator may set it in
# .env): tests must exercise the real gating logic by default; the kill-switch
# test monkeypatch.setenv's it explicitly.
os.environ["FINAQ_SKIP_FRESHNESS_PROBES"] = ""


@pytest.fixture(autouse=True)
def _isolated_state_db(tmp_path_factory, monkeypatch):
    """Point `data.state.DB_PATH` at a per-session tmp file so telemetry
    writes (from `_safe_node`, `invoke_with_telemetry`, etc.) never touch
    the real `data_cache/state.db`. Each session starts with a fresh DB."""
    from data import state as state_db

    test_db = tmp_path_factory.mktemp("state_db") / "test_state.db"
    monkeypatch.setattr(state_db, "DB_PATH", test_db)


@pytest.fixture(autouse=True)
def _no_real_vector_store_or_edgar_in_unit_tests(request, monkeypatch):
    """Unit tests must not touch Pinecone, the embeddings API, or the live
    EDGAR index. Opening an index or embedding text fails loudly so a test
    that needs them stubs `data.vectors._index` / `embed_texts` itself.

    The ingest-status probes are stubbed to "ingested, nothing stale" so the
    UI / Telegram / CIO paths exercise their happy path by default. Freshness
    tests monkeypatch these same seams themselves (test-level monkeypatch is
    applied after autouse fixtures, so it wins); tests marked `real_probes`
    get the real SQLite-backed probes against the isolated test DB.
    Integration and eval tests keep the real clients."""
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker(
        "eval"
    ):
        yield
        return
    from data import freshness as freshness_mod
    from data import vectors as vectors_mod

    def _no_pinecone(*args, **kwargs):
        raise AssertionError("unit tests must not open a Pinecone index — stub data.vectors._index")

    def _no_embeddings(*args, **kwargs):
        raise AssertionError("unit tests must not embed text — stub data.vectors.embed_texts")

    monkeypatch.setattr(vectors_mod, "_index", _no_pinecone)
    monkeypatch.setattr(vectors_mod, "embed_texts", _no_embeddings)
    if not request.node.get_closest_marker("real_probes"):
        monkeypatch.setattr(vectors_mod, "has_ticker", lambda ticker: True)
        monkeypatch.setattr(vectors_mod, "last_filings_by_type", lambda ticker: {})
    # data/freshness.py binds both probes at module top — patch its copies.
    monkeypatch.setattr(freshness_mod, "last_filings_by_type", lambda ticker: {})
    monkeypatch.setattr(
        freshness_mod, "latest_filing_dates", lambda ticker, forms=None: {}
    )
    yield
