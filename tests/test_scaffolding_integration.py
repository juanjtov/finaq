"""Step 1 integration smoke tests — hit real OpenRouter.

Requires a populated .env with a valid OPENROUTER_API_KEY.
Run via:  pytest -m integration tests/test_scaffolding_integration.py

Each test is parameterised over an *agent role* (MODEL_TRIAGE, MODEL_SYNTHESIS, ...)
rather than a model tier (haiku/sonnet/opus), so the tests stay valid even if you
swap any of the underlying models in .env.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


CHAT_ROLES = [
    "MODEL_TRIAGE",
    "MODEL_FUNDAMENTALS",
    "MODEL_FILINGS",
    "MODEL_NEWS",
    "MODEL_RISK",
    "MODEL_SYNTHESIS",
    "MODEL_ROUTER",
    "MODEL_ADHOC_THESIS",
    "MODEL_JUDGE",
]


@pytest.fixture(autouse=True)
def _require_api_key():
    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-v1-"):
        pytest.skip("OPENROUTER_API_KEY not set; populate .env to run integration tests")


@pytest.mark.parametrize("role_var", CHAT_ROLES)
def test_openrouter_chat_for_role(role_var):
    """Each configured chat-model role responds to a trivial prompt with non-empty text.
    Skips cleanly if the env var is missing or still the conftest stub — letting
    users add new MODEL_* vars to .env at their own pace."""
    from utils.openrouter import get_client

    model = os.environ.get(role_var, "")
    if not model or model == "test-stub-model":
        pytest.skip(f"{role_var} not set in .env (see .env.example)")
    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=10,
    )
    text = (resp.choices[0].message.content or "").strip()
    assert text, f"{role_var} ({model}) returned empty content"


def test_openrouter_embeddings_for_role():
    """The configured embeddings role returns a sensibly-shaped vector for a short input."""
    from utils.openrouter import get_client

    model = os.environ["MODEL_EMBEDDINGS"]
    resp = get_client().embeddings.create(model=model, input="hello world")
    vec = resp.data[0].embedding
    assert isinstance(vec, list) and len(vec) > 100, "embedding vector seems wrong shape"
    assert all(isinstance(x, float) for x in vec[:10]), "embeddings are not floats"
