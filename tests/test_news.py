"""Step 5c unit tests — News agent prompt assembly + failure paths.

Pure-logic, no network. Real Tavily + real LLM live in test_news_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.news import (
    NEWS_DAYS,
    _build_user_prompt,
    _company_name_for,
    _format_article,
    _strip_code_fences,
    run,
)
from utils.schemas import NewsOutput

THESES_DIR = Path(__file__).parents[1] / "theses"


def _fake_article(idx: int, title: str = "Sample article", **overrides) -> dict:
    return {
        "title": title,
        "url": f"https://example.com/article/{idx}",
        "content": "Some article body text.",
        "score": 0.85,
        "published_date": "2026-04-15",
        **overrides,
    }


# --- Article formatting -----------------------------------------------------


def test_format_article_renders_published_date_and_title():
    art = _fake_article(1, title="NVDA raises guide", published_date="2026-04-20")
    out = _format_article(1, art)
    assert "NVDA raises guide" in out
    assert "2026-04-20" in out
    assert "https://example.com/article/1" in out


def test_format_article_handles_missing_published_date():
    art = _fake_article(1, published_date=None)
    out = _format_article(1, art)
    assert "unknown" in out


def test_format_article_handles_missing_score():
    art = _fake_article(1, score=None)
    out = _format_article(1, art)
    assert "n/a" in out


# --- Prompt assembly --------------------------------------------------------


def test_build_user_prompt_includes_thesis_and_articles():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    articles = [_fake_article(1), _fake_article(2)]
    prompt = _build_user_prompt("NVDA", "NVIDIA Corporation", thesis, articles)
    assert "AS OF" in prompt
    assert "NVIDIA Corporation" in prompt
    assert "AI cake" in prompt
    assert "STRICT JSON" in prompt
    # Both articles rendered
    assert "article 1" in prompt
    assert "article 2" in prompt


def test_build_user_prompt_marks_empty_article_list():
    thesis = json.loads((THESES_DIR / "ai_cake.json").read_text())
    prompt = _build_user_prompt("NVDA", "NVIDIA", thesis, [])
    assert "no articles retrieved" in prompt.lower()


# --- _strip_code_fences -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"summary": "x"}', '{"summary": "x"}'),
        ('```json\n{"summary": "x"}\n```', '{"summary": "x"}'),
        ('```\n{"summary": "x"}\n```', '{"summary": "x"}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# --- _company_name_for ------------------------------------------------------


def test_company_name_falls_back_to_ticker_when_yfinance_fails(monkeypatch):
    from agents import news as n

    def boom(_):
        raise RuntimeError("yfinance outage")

    monkeypatch.setattr(n, "get_financials", boom)
    assert _company_name_for("NVDA") == "NVDA"


def test_company_name_uses_long_name_when_present(monkeypatch):
    from agents import news as n

    monkeypatch.setattr(n, "get_financials", lambda _: {"info": {"longName": "NVIDIA Corporation"}})
    assert _company_name_for("NVDA") == "NVIDIA Corporation"


def test_company_name_falls_back_to_short_name(monkeypatch):
    from agents import news as n

    monkeypatch.setattr(n, "get_financials", lambda _: {"info": {"shortName": "NVIDIA"}})
    assert _company_name_for("NVDA") == "NVIDIA"


# --- run() failure paths ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_stale_news_message_when_tavily_empty(monkeypatch):
    """Tavily returned no articles — output is schema-valid with a clear staleness flag."""
    from agents import news as n

    monkeypatch.setattr(n, "_company_name_for", lambda t: "Stub Co.")
    monkeypatch.setattr(n, "search_news", lambda *a, **kw: [])

    state = {"ticker": "ZZZZ", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])
    assert "STALE NEWS" in out.summary
    assert any("no articles" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_falls_back_when_llm_fails(monkeypatch):
    """LLM raises but Tavily succeeded — output must still be schema-valid."""
    from agents import news as n

    monkeypatch.setattr(n, "_company_name_for", lambda t: "Stub Co.")
    monkeypatch.setattr(n, "search_news", lambda *a, **kw: [_fake_article(1)])

    def boom(*a, **kw):
        raise RuntimeError("simulated openrouter outage")

    monkeypatch.setattr(n, "_call_llm", boom)

    state = {"ticker": "STUB", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])
    assert any("llm" in e for e in out.errors)
    # Tavily error should NOT be in the list — only LLM
    assert not any("tavily" in e for e in out.errors)


@pytest.mark.asyncio
async def test_run_propagates_llm_output_when_call_succeeds(monkeypatch):
    from agents import news as n

    monkeypatch.setattr(n, "_company_name_for", lambda t: "Stub Co.")
    monkeypatch.setattr(n, "search_news", lambda *a, **kw: [_fake_article(1)])

    fake_response = {
        "summary": "[stub] thesis-aware news synthesis",
        "catalysts": [
            {
                "title": "stub catalyst",
                "summary": "thesis-relevant move",
                "sentiment": "bull",
                "url": "https://example.com/c",
                "as_of": "2026-04-20",
            }
        ],
        "concerns": [
            {
                "title": "stub concern",
                "summary": "thesis-relevant risk",
                "sentiment": "bear",
                "url": "https://example.com/r",
                "as_of": "2026-04-21",
            }
        ],
        "evidence": [
            {
                "source": "tavily",
                "url": "https://example.com/c",
                "excerpt": "key phrase",
                "as_of": "2026-04-20",
                "note": "matters because thesis",
            }
        ],
    }
    monkeypatch.setattr(n, "_call_llm", lambda *a, **kw: fake_response)

    state = {"ticker": "NVDA", "thesis": json.loads((THESES_DIR / "ai_cake.json").read_text())}
    result = await run(state)
    out = NewsOutput.model_validate(result["news"])
    assert "[stub]" in out.summary
    assert out.catalysts[0].as_of == "2026-04-20"
    assert out.concerns[0].sentiment == "bear"
    assert out.evidence[0].as_of == "2026-04-20"
    assert out.errors == []


def test_news_default_window_is_90_days_per_spec():
    """CLAUDE.md §9.3: 'Tavily search ... over the past 90 days'."""
    assert NEWS_DAYS == 90
