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


@pytest.fixture(autouse=True)
def _isolated_state_db(tmp_path_factory, monkeypatch):
    """Point `data.state.DB_PATH` at a per-session tmp file so telemetry
    writes (from `_safe_node`, `invoke_with_telemetry`, etc.) never touch
    the real `data_cache/state.db`. Each session starts with a fresh DB."""
    from data import state as state_db

    test_db = tmp_path_factory.mktemp("state_db") / "test_state.db"
    monkeypatch.setattr(state_db, "DB_PATH", test_db)
