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

# Same treatment for the freshness-probe kill-switch (may be set in the
# user's .env while the chromadb segfault is unfixed — POSTPONED §2): tests
# must exercise the real gating logic by default; the kill-switch test
# monkeypatch.setenv's it explicitly.
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
def _no_real_chroma_or_edgar_in_unit_tests(request, monkeypatch):
    """Unit tests must not touch the real ChromaDB corpus or the live EDGAR
    index. Beyond hermeticity, chromadb's Rust client segfaults under pytest
    on macOS when unit tests reach `data_cache/chroma/` (observed at three
    call sites: the drill-time freshness gate in agents/filings.py and
    data/telegram.py, and the dashboard ingest banner / Mission Control
    freshness sweep under AppTest) — a segfault kills the whole session, so
    it can't even be caught per-test.

    Stubs the probe layer only; `check_ingest_freshness`'s real logic stays
    testable because freshness tests monkeypatch these same seams themselves
    (test-level monkeypatch is applied after autouse fixtures, so it wins).
    Integration and eval tests keep the real clients."""
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker(
        "eval"
    ):
        yield
        return
    from data import chroma as chroma_mod
    from data import freshness as freshness_mod

    monkeypatch.setattr(chroma_mod, "has_ticker", lambda ticker: True)
    monkeypatch.setattr(chroma_mod, "last_filings_by_type", lambda ticker: {})
    # data/freshness.py binds both probes at module top — patch its copies.
    monkeypatch.setattr(freshness_mod, "last_filings_by_type", lambda ticker: {})
    monkeypatch.setattr(
        freshness_mod, "latest_filing_dates", lambda ticker, forms=None: {}
    )
    yield
