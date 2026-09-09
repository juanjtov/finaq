"""`data/finnhub.py` — slicing, relevance filter, sampling, bodies, backtest cache.

Pure-logic: `httpx.get` is faked. The real endpoint is exercised by
`pytest -m integration tests/test_news_integration.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from data import finnhub

TODAY = datetime.now(UTC).date()  # the wrapper's "today" is the UTC date


def _ts(day: date, hour: int = 12) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp())


def _item(id_: int, day: date, headline: str, summary: str = "", source: str = "Yahoo") -> dict:
    return {
        "id": id_,
        "datetime": _ts(day),
        "headline": headline,
        "summary": summary,
        "source": source,
        "url": f"https://finnhub.io/api/news?id={id_}",
    }


class _Resp:
    def __init__(self, *, payload=None, text="", url="", status_code=200):
        self._payload = payload
        self.text = text
        self.url = url
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_http(monkeypatch):
    """Fake `httpx.get`: the Finnhub endpoint serves `state["items"]` filtered
    by from/to (newest first, capped like the real API); any other URL is an
    article page. Records every call."""
    state = {
        "items": [],
        "calls": [],
        "article_html": "<p>" + "x" * 80 + "</p>",
        "fail_articles": False,
        "status": 200,
    }

    def fake_get(url, params=None, headers=None, **kwargs):
        if url == finnhub.FINNHUB_URL:
            state["calls"].append({**params, "headers": headers or {}})
            if state["status"] != 200:
                return _Resp(payload=None, status_code=state["status"])
            lo, hi = date.fromisoformat(params["from"]), date.fromisoformat(params["to"])
            hits = [
                i
                for i in state["items"]
                if lo <= datetime.fromtimestamp(i["datetime"], UTC).date() <= hi
            ]
            hits.sort(key=lambda i: -i["datetime"])
            return _Resp(payload=hits[:250])
        state["calls"].append({"article": url})
        if state["fail_articles"]:
            raise finnhub.httpx.ConnectError("boom")
        return _Resp(
            text=state["article_html"], url=url.replace("finnhub.io/api/news?id=", "pub.example/")
        )

    monkeypatch.setattr(finnhub.httpx, "get", fake_get)
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(finnhub, "BACKTEST_CACHE_DIR", None)  # set per-test when needed
    return state


# --- Key handling -----------------------------------------------------------


def test_missing_key_returns_empty_without_calling(fake_http, monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY")
    assert finnhub.search_news("NVDA", "NVIDIA Corporation") == []
    assert fake_http["calls"] == []


# --- Slicing ----------------------------------------------------------------


def test_slice_bounds_are_log_spaced_and_cover_the_window():
    bounds = finnhub._slice_bounds(TODAY, 90)
    assert len(bounds) == 6
    assert bounds[0] == (TODAY - timedelta(days=3), TODAY)
    assert bounds[-1] == (TODAY - timedelta(days=90), TODAY - timedelta(days=60))
    # consecutive slices share their boundary day — nothing falls between them
    for (newer_start, _), (_, older_end) in zip(bounds, bounds[1:], strict=False):
        assert newer_start == older_end


def test_slice_bounds_clip_to_short_windows():
    assert finnhub._slice_bounds(TODAY, 14) == [
        (TODAY - timedelta(days=3), TODAY),
        (TODAY - timedelta(days=7), TODAY - timedelta(days=3)),
        (TODAY - timedelta(days=14), TODAY - timedelta(days=7)),
    ]
    assert finnhub._slice_bounds(TODAY, 2) == [(TODAY - timedelta(days=2), TODAY)]


def test_fetch_window_makes_one_call_per_slice_and_dedupes_boundary_items(fake_http):
    boundary = TODAY - timedelta(days=3)  # belongs to slices 0 and 1
    fake_http["items"] = [
        _item(1, TODAY, "a"),
        _item(2, boundary, "b"),
        _item(3, TODAY - timedelta(days=40), "c"),
    ]
    out = finnhub._fetch_window("NVDA", TODAY, 90, "k")
    assert len(fake_http["calls"]) == 6
    assert sorted(i["id"] for i in out) == [1, 2, 3]
    assert all(c["symbol"] == "NVDA" for c in fake_http["calls"])


def test_api_key_travels_in_a_header_not_the_url(fake_http):
    fake_http["items"] = []
    finnhub.search_news("NVDA", "NVIDIA", with_body=False)
    for call in fake_http["calls"]:
        assert call["headers"] == {"X-Finnhub-Token": "test-key"}
        assert "token" not in call


def test_client_error_is_not_retried_and_raises(fake_http):
    fake_http["status"] = 401
    with pytest.raises(RuntimeError, match="HTTP 401"):
        finnhub.search_news("NVDA", "NVIDIA", days=2, with_body=False)
    assert len(fake_http["calls"]) == 1  # one slice, no tenacity retries on 4xx


def test_fetch_window_keeps_items_without_ids(fake_http):
    a = _item(0, TODAY, "a")
    b = _item(0, TODAY, "b")
    a.pop("id")
    b.pop("id")
    b["url"] = "https://finnhub.io/api/news?id=other"
    fake_http["items"] = [a, b]
    assert len(finnhub._fetch_window("NVDA", TODAY, 2, "k")) == 2


# --- Relevance filter -------------------------------------------------------


def test_mention_pattern_matches_ticker_or_company_phrase():
    pat = finnhub._mention_pattern("NVDA", "NVIDIA Corporation")
    assert pat.search("Nvidia beats estimates")
    assert pat.search("Chip names rally (NVDA, AMD)")
    assert pat.search("Buy NVDA before earnings")
    assert not pat.search("Why Braze stock tumbled on Tuesday")
    assert not pat.search("nvda-adjacent")  # lowercase ticker inside a token is not a mention


def test_mention_pattern_short_tickers_need_citation_form_or_name_phrase():
    pat = finnhub._mention_pattern("NU", "Nu Holdings Ltd.")
    assert pat.search("Nubank (NU) posts record quarter")
    assert pat.search("Nu Holdings reports Q2")
    assert pat.search("Wolfe maintains Outperform on nu holdings")
    assert pat.search("NYSE:NU jumps")
    assert not pat.search("NUE steel prices climb")
    assert not pat.search("Menu changes at Wendy's")
    assert not pat.search("Two NU students win prize")  # bare 2-char ticker is not enough
    pat = finnhub._mention_pattern("AI", "C3.ai, Inc.")
    assert not pat.search("Chipmakers rally as AI demand surges")
    assert pat.search("C3.ai (NYSE: AI) lands DoD contract")
    assert pat.search("c3.ai shares slide")
    pat = finnhub._mention_pattern("S", "SentinelOne, Inc.")
    assert not pat.search("S&P 500 closes at record high")
    assert pat.search("SentinelOne beats on ARR")


def test_mention_pattern_uses_two_name_words_against_generic_first_words():
    pat = finnhub._mention_pattern("QSR", "Restaurant Brands International Inc.")
    assert not pat.search("Restaurant stocks fall on labour costs")
    assert pat.search("Restaurant Brands raises dividend")
    assert pat.search("QSR posts same-store sales growth")
    berkshire = finnhub._mention_pattern("BRK.B", "Berkshire Hathaway Inc.")
    assert berkshire.search("Berkshire Hathaway (BRK.B) buys")


def test_search_drops_untagged_and_duplicate_headlines(fake_http):
    fake_http["items"] = [
        _item(1, TODAY, "Nvidia raises guide", "AI demand"),
        _item(2, TODAY, "Nvidia raises guide!", "same story, other outlet"),
        _item(3, TODAY, "Market wrap: Dow slides", "Nothing about the company"),
        _item(4, TODAY, "Three stocks to watch", "NVDA among them"),
        _item(
            5, TODAY, "輝達財報亮眼", "NVIDIA 第二季營收創新高"
        ),  # non-ASCII title survives de-dup
    ]
    out = finnhub.search_news("NVDA", "NVIDIA Corporation", with_body=False)
    assert [a["title"] for a in out] == [
        "Nvidia raises guide",
        "Three stocks to watch",
        "輝達財報亮眼",
    ]


# --- Sampling ---------------------------------------------------------------


def _busy_ticker_items() -> list[dict]:
    """Ten NVDA articles per day for the whole quarter."""
    items, n = [], 0
    for back in range(0, 91):
        for _ in range(10):
            n += 1
            items.append(_item(n, TODAY - timedelta(days=back), f"NVDA story {n}"))
    return items


def test_busy_ticker_sample_spans_every_slice_newest_first(fake_http):
    fake_http["items"] = _busy_ticker_items()
    out = finnhub.search_news("NVDA", "NVIDIA Corporation", with_body=False)
    assert len(out) == 15
    days_back = [(TODAY - date.fromisoformat(a["published_date"][:10])).days for a in out]
    assert days_back == sorted(days_back)  # newest first
    # 3/3/3/2/2/2 across 0-3 / 3-7 / 7-14 / 14-30 / 30-60 / 60-90 days back
    assert sum(d <= 3 for d in days_back) == 3
    assert sum(3 < d <= 7 for d in days_back) == 3
    assert sum(30 < d <= 60 for d in days_back) == 2
    assert sum(60 < d <= 90 for d in days_back) == 2


def test_thin_ticker_backfills_from_leftovers(fake_http):
    fake_http["items"] = [
        _item(n, TODAY - timedelta(days=n % 2), f"COUR update {n}") for n in range(1, 9)
    ]
    out = finnhub.search_news("COUR", "Coursera, Inc.", max_results=6, with_body=False)
    assert len(out) == 6  # slice quota alone would allow only 3 from the newest slice
    assert all("COUR" in a["title"] for a in out)


def test_cio_window_shape(fake_http):
    fake_http["items"] = _busy_ticker_items()
    out = finnhub.search_news("NVDA", "NVIDIA", days=14, max_results=8, with_body=False)
    assert len(out) == 8
    assert len(fake_http["calls"]) == 3
    assert all(not c.get("article") for c in fake_http["calls"])
    assert all(a["url"].startswith("https://finnhub.io/api/news?id=") for a in out)


def test_output_shape(fake_http):
    fake_http["items"] = [_item(1, TODAY, "Nvidia wins", "big deal", source="Benzinga")]
    (art,) = finnhub.search_news("NVDA", "NVIDIA Corporation", with_body=False)
    assert art == {
        "title": "Nvidia wins",
        "url": "https://finnhub.io/api/news?id=1",
        "content": "big deal",
        "source": "Benzinga",
        "score": None,
        "published_date": datetime.fromtimestamp(_ts(TODAY), UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --- Article bodies ---------------------------------------------------------


def test_with_body_extends_content_and_resolves_publisher_url(fake_http):
    fake_http["items"] = [_item(1, TODAY, "Nvidia wins", "summary")]
    fake_http["article_html"] = (
        "<p>short</p><p>" + "Long paragraph about NVIDIA's quarter. " * 5 + "</p>"
    )
    (art,) = finnhub.search_news("NVDA", "NVIDIA Corporation")
    assert art["url"] == "https://pub.example/1"
    assert art["content"].startswith("summary Long paragraph")
    assert len(art["content"]) <= len("summary ") + finnhub.BODY_MAX_CHARS


def test_body_fetch_failure_keeps_summary_and_finnhub_url(fake_http):
    fake_http["items"] = [_item(1, TODAY, "Nvidia wins", "summary")]
    fake_http["fail_articles"] = True
    (art,) = finnhub.search_news("NVDA", "NVIDIA Corporation")
    assert art["url"] == "https://finnhub.io/api/news?id=1"
    assert art["content"] == "summary"


def test_stub_page_keeps_summary_and_finnhub_url(fake_http):
    """A consent page or paywall stub yields no more text than the summary —
    keep the redirect link rather than citing the stub."""
    fake_http["items"] = [_item(1, TODAY, "Nvidia wins", "A long summary " * 10)]
    fake_http["article_html"] = "<p>Please accept cookies to continue reading this article.</p>"
    (art,) = finnhub.search_news("NVDA", "NVIDIA Corporation")
    assert art["url"] == "https://finnhub.io/api/news?id=1"
    assert art["content"] == "A long summary " * 10


# --- Backtest mode ----------------------------------------------------------


def test_backtest_window_ends_at_as_of_and_is_cached(fake_http, tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub, "BACKTEST_CACHE_DIR", tmp_path)
    as_of = date(2025, 9, 5)
    fake_http["items"] = [
        _item(1, as_of - timedelta(days=10), "Intel pre as-of"),
        _item(2, as_of + timedelta(days=1), "Intel post as-of"),  # served if a slice asked for it
    ]
    out = finnhub.search_news("INTC", "Intel Corporation", as_of="2025-09-05", with_body=False)
    assert [a["title"] for a in out] == ["Intel pre as-of"]
    assert max(c["to"] for c in fake_http["calls"]) == "2025-09-05"
    assert min(c["from"] for c in fake_http["calls"]) == "2025-06-07"  # 90 days back

    cache = tmp_path / "INTC__as_of_2025-09-05__headlines.json"  # keyed by with_body too
    assert json.loads(cache.read_text()) == out
    n_calls = len(fake_http["calls"])
    again = finnhub.search_news("INTC", "Intel Corporation", as_of="2025-09-05", with_body=False)
    assert again == out
    assert len(fake_http["calls"]) == n_calls  # cache hit: no network


def test_backtest_empty_result_is_cached_only_inside_the_history_horizon(
    fake_http, tmp_path, monkeypatch
):
    monkeypatch.setattr(finnhub, "BACKTEST_CACHE_DIR", tmp_path)
    fake_http["items"] = []
    today = datetime.now(UTC).date()
    recent = (today - timedelta(days=30)).isoformat()
    ancient = (today - timedelta(days=finnhub.HISTORY_DAYS + 30)).isoformat()
    assert finnhub.search_news("INTC", "Intel", as_of=recent, with_body=False) == []
    assert finnhub.search_news("INTC", "Intel", as_of=ancient, with_body=False) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"INTC__as_of_{recent}__headlines.json"]


def test_backtest_drops_articles_outside_window_defensively(fake_http, tmp_path, monkeypatch):
    """If the API ever ignored `to`, a post-as_of article must still not leak."""
    monkeypatch.setattr(finnhub, "BACKTEST_CACHE_DIR", tmp_path)
    leaked = _item(9, date(2025, 9, 20), "Intel leaked future")
    monkeypatch.setattr(finnhub, "_fetch_window", lambda *a, **k: [leaked])
    assert finnhub.search_news("INTC", "Intel", as_of="2025-09-05", with_body=False) == []


def test_production_path_ends_today_and_does_not_cache(fake_http, tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub, "BACKTEST_CACHE_DIR", tmp_path)
    fake_http["items"] = []
    finnhub.search_news("INTC", "Intel")
    assert max(c["to"] for c in fake_http["calls"]) == datetime.now(UTC).date().isoformat()
    assert list(tmp_path.iterdir()) == []
