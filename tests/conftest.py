"""Test fixtures and env-stub setup.

Loads the real .env first so integration tests see the user's keys + model
strings, then fills in non-empty placeholders for any MODEL_* var still
missing so unit tests work without a populated .env.
"""

from __future__ import annotations

import os

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
    "MODEL_EMBEDDINGS",
)

for _name in _MODEL_STUB_VARS:
    os.environ.setdefault(_name, _STUB)
