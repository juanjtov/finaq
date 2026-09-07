"""`data/tavily.py` — `search_depth` passthrough.

The CIO planner only reads headlines, so it asks for "basic" depth
(1 Tavily credit) instead of the News agent's "advanced" (2 credits).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-test-key")
    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": []}

    monkeypatch.setattr("tavily.TavilyClient", _FakeClient)
    return captured


def test_search_depth_defaults_to_advanced(fake_tavily):
    from data import tavily

    tavily.search_news("NVDA", "NVIDIA")
    assert fake_tavily["search_depth"] == "advanced"


def test_search_depth_basic_passes_through_with_cio_window(fake_tavily):
    from data import tavily

    tavily.search_news("NVDA", "NVIDIA", days=14, max_results=8, search_depth="basic")
    assert fake_tavily["search_depth"] == "basic"
    assert fake_tavily["days"] == 14
    assert fake_tavily["max_results"] == 8
    assert fake_tavily["query"] == "NVDA NVIDIA"
