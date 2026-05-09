"""Per-data-source `as_of` correctness tests for the backtest harness (Step B1).

The CLI under test:
  - `data.yfin.get_financials(ticker, as_of=...)` — yfinance filtering
  - `data.edgar._existing_filings(ticker, kind, as_of=...)` — file-mtime / SGML-header gate
  - `data.chroma._build_where_clause(..., as_of=...)` — ChromaDB metadata filter
  - `data.treasury.get_10y_treasury_yield(as_of=...)` — historical FRED-style lookup

Each test proves the gate actually drops post-as_of data and that the
production path (`as_of=None`) is unchanged.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest


# --- yfinance ---------------------------------------------------------------


def test_yfin_filter_df_drops_post_as_of_rows():
    """`_filter_df_by_as_of` (the price-history path, transpose=False) must
    drop rows whose index date is after as_of."""
    import pandas as pd

    from data.yfin import _filter_df_by_as_of

    df = pd.DataFrame(
        {"Close": [100, 101, 102, 103, 104]},
        index=pd.to_datetime(
            ["2025-09-01", "2025-09-03", "2025-09-05", "2025-09-08", "2025-09-15"]
        ),
    )
    out = _filter_df_by_as_of(df, date(2025, 9, 5))
    assert len(out) == 3
    assert out.index.max() == pd.Timestamp("2025-09-05")


def test_yfin_filter_df_drops_post_as_of_columns_for_financials():
    """`_filter_df_by_as_of` (the financial-statement path, transpose=True)
    must drop COLUMNS whose date is after as_of, since financial DataFrames
    have line items as rows and dates as columns."""
    import pandas as pd

    from data.yfin import _filter_df_by_as_of

    df = pd.DataFrame(
        [[1, 2, 3, 4]],
        index=["TotalRevenue"],
        columns=pd.to_datetime(["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"]),
    )
    out = _filter_df_by_as_of(df, date(2025, 9, 5), transpose=True)
    # Only the first three columns should survive (Sep 30 > Sep 5).
    assert out.shape[1] == 3
    assert pd.Timestamp("2025-09-30") not in out.columns


def test_yfin_filter_df_handles_tz_aware_index():
    """yfinance occasionally returns tz-aware timestamps; the filter must
    not crash and must compare against the naive as_of correctly."""
    import pandas as pd

    from data.yfin import _filter_df_by_as_of

    df = pd.DataFrame(
        {"Close": [100, 101, 102]},
        index=pd.to_datetime(
            ["2025-09-01 09:30:00", "2025-09-05 16:00:00", "2025-09-10 16:00:00"]
        ).tz_localize("America/New_York"),
    )
    out = _filter_df_by_as_of(df, date(2025, 9, 5))
    assert len(out) == 2  # Sep 1 and Sep 5 land at or before as_of


def test_yfin_get_financials_uses_backtest_cache_path(tmp_path, monkeypatch):
    """Backtest mode (as_of=...) writes to `{ticker}__as_of_{date}.json`,
    NOT the production `{ticker}.json` cache."""
    from data import yfin

    monkeypatch.setattr(yfin, "CACHE_DIR", tmp_path)

    fake = {
        "fetched_at": "2026-05-08",
        "as_of_date": "2025-09-05",
        "price_history_5y": {},
        "income_stmt": {},
        "balance_sheet": {},
        "cash_flow": {},
        "info": {},
    }
    monkeypatch.setattr(yfin, "_fetch_from_yfinance", lambda ticker, *, as_of=None: fake)

    yfin.get_financials("INTC", as_of="2025-09-05")
    assert (tmp_path / "INTC__as_of_2025-09-05.json").exists()
    assert not (tmp_path / "INTC.json").exists(), (
        "backtest cache must NOT clobber the production cache file"
    )


def test_yfin_get_financials_production_path_unchanged(tmp_path, monkeypatch):
    """`as_of=None` (the production default) keeps writing to the existing
    cache filename. Regression guard for backwards compatibility."""
    from data import yfin

    monkeypatch.setattr(yfin, "CACHE_DIR", tmp_path)
    fake = {
        "fetched_at": "2026-05-08",
        "as_of_date": None,
        "price_history_5y": {},
        "income_stmt": {},
        "balance_sheet": {},
        "cash_flow": {},
        "info": {},
    }
    monkeypatch.setattr(yfin, "_fetch_from_yfinance", lambda ticker, *, as_of=None: fake)

    yfin.get_financials("INTC")
    assert (tmp_path / "INTC.json").exists()


# --- EDGAR ------------------------------------------------------------------


def test_edgar_existing_filings_drops_post_as_of_files(tmp_path, monkeypatch):
    """`_existing_filings(ticker, kind, as_of=...)` filters by
    SGML-header `FILED AS OF DATE`. Filings dated after as_of must be excluded."""
    from data import edgar

    monkeypatch.setattr(edgar, "EDGAR_DIR", tmp_path)
    base = tmp_path / "sec-edgar-filings" / "INTC" / "10-K"
    base.mkdir(parents=True)

    filings = [
        ("0001-25-100", "20250715"),  # before as_of
        ("0001-25-200", "20250901"),  # before as_of
        ("0001-25-300", "20251115"),  # AFTER as_of (must be dropped)
    ]
    for accession, filed in filings:
        (base / accession).mkdir()
        (base / accession / "full-submission.txt").write_text(
            f"<SEC-DOCUMENT>\nFILED AS OF DATE:\t{filed}\n<TYPE>10-K\n"
        )

    paths_no_filter = edgar._existing_filings("INTC", "10-K")
    assert len(paths_no_filter) == 3, "production mode keeps all on-disk filings"

    paths_filtered = edgar._existing_filings("INTC", "10-K", as_of="2025-09-05")
    assert len(paths_filtered) == 2, "post-as_of filing must be excluded"
    accessions = {p.parent.name for p in paths_filtered}
    assert "0001-25-300" not in accessions


def test_edgar_existing_filings_drops_undated_in_backtest_mode(tmp_path, monkeypatch):
    """A filing whose SGML header is missing or unparseable must be excluded
    in backtest mode (we can't prove it's pre-as_of). Production mode keeps it."""
    from data import edgar

    monkeypatch.setattr(edgar, "EDGAR_DIR", tmp_path)
    base = tmp_path / "sec-edgar-filings" / "INTC" / "10-K"
    base.mkdir(parents=True)

    (base / "0001-25-100").mkdir()
    (base / "0001-25-100" / "full-submission.txt").write_text(
        "<SEC-DOCUMENT>\nFILED AS OF DATE:\t20250715\n"
    )
    (base / "0001-25-noheader").mkdir()
    (base / "0001-25-noheader" / "full-submission.txt").write_text(
        "<SEC-DOCUMENT>\n(no filed-as-of-date header)\n"
    )

    prod = edgar._existing_filings("INTC", "10-K")
    bt = edgar._existing_filings("INTC", "10-K", as_of="2025-09-05")

    assert len(prod) == 2, "production keeps undated filings"
    assert len(bt) == 1, "backtest drops the undated filing"


# --- ChromaDB ---------------------------------------------------------------


def test_chroma_where_clause_includes_filed_date_lte_when_as_of_set():
    """`_build_where_clause(as_of=...)` must produce a ChromaDB filter that
    restricts to chunks with `filed_date <= as_of`."""
    from data.chroma import _build_where_clause

    where = _build_where_clause("INTC", "1A", as_of="2025-09-05")
    # Three conditions: ticker + item_code + filed_date
    assert "$and" in where
    conds = where["$and"]
    assert {"ticker": "INTC"} in conds
    assert {"item_code": "1A"} in conds
    assert {"filed_date": {"$lte": "2025-09-05"}} in conds


def test_chroma_where_clause_no_as_of_unchanged():
    """Production path (`as_of=None`) must NOT include any filed_date filter."""
    from data.chroma import _build_where_clause

    where = _build_where_clause("INTC", "1A")
    assert "$and" in where
    serialised = json.dumps(where)
    assert "filed_date" not in serialised
    assert "$lte" not in serialised


def test_chroma_where_clause_only_as_of_returns_single_filter():
    """With ticker=None and item_filter=None but as_of set, the clause is
    just the date filter — no $and wrapper."""
    from data.chroma import _build_where_clause

    where = _build_where_clause(None, None, as_of="2025-09-05")
    assert where == {"filed_date": {"$lte": "2025-09-05"}}


# --- Treasury ---------------------------------------------------------------


def test_treasury_backtest_cache_path_is_keyed_by_as_of(tmp_path, monkeypatch):
    """Backtest mode writes to `treasury__as_of_{date}.json`, never clobbers
    the production `treasury.json`."""
    from data import treasury

    monkeypatch.setattr(treasury, "CACHE_PATH", tmp_path / "treasury.json")
    monkeypatch.setattr(treasury, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(treasury, "_fetch_yield", lambda as_of=None: 0.045)

    treasury.get_10y_treasury_yield(as_of="2025-09-05")
    assert (tmp_path / "treasury__as_of_2025-09-05.json").exists()
    assert not (tmp_path / "treasury.json").exists()


def test_treasury_production_path_unchanged(tmp_path, monkeypatch):
    from data import treasury

    monkeypatch.setattr(treasury, "CACHE_PATH", tmp_path / "treasury.json")
    monkeypatch.setattr(treasury, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(treasury, "_fetch_yield", lambda as_of=None: 0.043)

    y = treasury.get_10y_treasury_yield()
    assert y == 0.043
    assert (tmp_path / "treasury.json").exists()


def test_treasury_passes_as_of_to_fetch(tmp_path, monkeypatch):
    """The historical fetch must receive the parsed `date` object."""
    from data import treasury

    monkeypatch.setattr(treasury, "CACHE_PATH", tmp_path / "treasury.json")
    monkeypatch.setattr(treasury, "CACHE_DIR", tmp_path)

    captured: dict = {}

    def _stub_fetch(as_of=None):
        captured["as_of"] = as_of
        return 0.041

    monkeypatch.setattr(treasury, "_fetch_yield", _stub_fetch)

    treasury.get_10y_treasury_yield(as_of="2025-09-05")
    assert captured["as_of"] == date(2025, 9, 5)


def test_treasury_fallback_used_on_fetch_failure(tmp_path, monkeypatch):
    from data import treasury

    monkeypatch.setattr(treasury, "CACHE_PATH", tmp_path / "treasury.json")
    monkeypatch.setattr(treasury, "CACHE_DIR", tmp_path)

    def _boom(as_of=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(treasury, "_fetch_yield", _boom)

    y = treasury.get_10y_treasury_yield(as_of="2025-09-05")
    assert y == treasury.DEFAULT_FALLBACK
    # On failure we should NOT write a cache file (next call should retry, not
    # learn the wrong answer).
    assert not (tmp_path / "treasury__as_of_2025-09-05.json").exists()


# --- FinaqState shape regression --------------------------------------------


def test_finaq_state_accepts_as_of_date_field():
    """`FinaqState` is total=False; verify our new optional field round-trips."""
    from utils.state import FinaqState

    state: FinaqState = {"ticker": "INTC", "as_of_date": "2025-09-05"}
    assert state["as_of_date"] == "2025-09-05"

    # Production path: as_of_date may be omitted entirely.
    state2: FinaqState = {"ticker": "INTC"}
    assert state2.get("as_of_date") is None


# --- Tavily news as_of (Step B2) -------------------------------------------


def test_tavily_search_news_passes_start_end_dates_to_client(tmp_path, monkeypatch):
    """Backtest mode (`as_of=...`) calls Tavily with start_date + end_date,
    not the production `days` parameter."""
    from data import tavily

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-test-key")
    monkeypatch.setattr(tavily, "BACKTEST_CACHE_DIR", tmp_path)

    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            captured.update(kwargs)
            return {
                "results": [
                    {
                        "title": "Pre as-of headline",
                        "url": "https://example.com/a",
                        "content": "...",
                        "score": 0.9,
                        "published_date": "2025-08-15T10:00:00Z",
                    }
                ]
            }

    monkeypatch.setattr("tavily.TavilyClient", _FakeClient)

    out = tavily.search_news("INTC", "Intel Corporation", days=90, as_of="2025-09-05")

    # Backtest path: `start_date` / `end_date` substituted for `days`.
    assert "start_date" in captured
    assert "end_date" in captured
    assert "days" not in captured
    assert captured["end_date"] == "2025-09-05"
    assert captured["start_date"] == "2025-06-07"  # 2025-09-05 minus 90 days
    assert len(out) == 1


def test_tavily_search_news_filters_post_as_of_articles(tmp_path, monkeypatch):
    """Defence in depth: even if Tavily returns an article dated AFTER as_of,
    the wrapper drops it."""
    from data import tavily

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-test-key")
    monkeypatch.setattr(tavily, "BACKTEST_CACHE_DIR", tmp_path)

    class _FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            return {
                "results": [
                    {  # before as_of — kept
                        "title": "Aug headline",
                        "url": "https://example.com/aug",
                        "content": "x",
                        "score": 0.9,
                        "published_date": "2025-08-15T10:00:00Z",
                    },
                    {  # after as_of — dropped
                        "title": "Nov headline",
                        "url": "https://example.com/nov",
                        "content": "y",
                        "score": 0.85,
                        "published_date": "2025-11-22T14:00:00Z",
                    },
                    {  # missing date — dropped (conservative posture)
                        "title": "Undated headline",
                        "url": "https://example.com/u",
                        "content": "z",
                        "score": 0.8,
                    },
                ]
            }

    monkeypatch.setattr("tavily.TavilyClient", _FakeClient)

    out = tavily.search_news("INTC", "Intel", days=90, as_of="2025-09-05")
    assert len(out) == 1
    assert out[0]["title"] == "Aug headline"


def test_tavily_search_news_caches_backtest_results(tmp_path, monkeypatch):
    """First call hits the API + writes a cache file; second call reads
    from cache without hitting the API."""
    from data import tavily

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-test-key")
    monkeypatch.setattr(tavily, "BACKTEST_CACHE_DIR", tmp_path)

    api_calls = {"n": 0}

    class _FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            api_calls["n"] += 1
            return {"results": []}

    monkeypatch.setattr("tavily.TavilyClient", _FakeClient)

    tavily.search_news("INTC", "Intel", as_of="2025-09-05")
    assert api_calls["n"] == 1
    assert (tmp_path / "INTC__as_of_2025-09-05.json").exists()

    # Second call — must hit the cache, not the API.
    tavily.search_news("INTC", "Intel", as_of="2025-09-05")
    assert api_calls["n"] == 1, "second call should hit cache, not Tavily"


def test_tavily_search_news_production_path_unchanged(monkeypatch):
    """`as_of=None` preserves the existing production behaviour: passes
    `days` (not `start_date`/`end_date`), no caching, no filtering."""
    from data import tavily

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake-test-key")
    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": [
                {"title": "x", "url": "u", "content": "c", "score": 0.5,
                 "published_date": "2026-04-01T00:00:00Z"},
            ]}

    monkeypatch.setattr("tavily.TavilyClient", _FakeClient)

    tavily.search_news("INTC", "Intel")
    assert captured.get("days") == 90
    assert "start_date" not in captured
    assert "end_date" not in captured


# --- Agent plumbing — fundamentals, filings, news, risk, synthesis ---------


def test_agents_thread_as_of_to_data_layer(monkeypatch):
    """End-to-end check that each agent's `run` reads `state["as_of_date"]`
    and forwards it to its data-layer call. We stub out the LLM and the
    data-layer call sites and assert they got `as_of` in kwargs."""
    import asyncio

    captured: dict[str, dict] = {}

    # --- Fundamentals
    from agents import fundamentals as fa

    def _stub_get_fin(ticker, *, as_of=None):
        captured["fundamentals"] = {"ticker": ticker, "as_of": as_of}
        return {"price_history_5y": {}, "income_stmt": {}, "balance_sheet": {},
                "cash_flow": {}, "info": {"longName": "Intel Corp"}}

    monkeypatch.setattr(fa, "get_financials", _stub_get_fin)
    monkeypatch.setattr(fa, "compute_kpis", lambda f: {})  # short-circuit LLM path

    asyncio.run(fa.run({"ticker": "INTC", "thesis": {}, "as_of_date": "2025-09-05"}))
    assert captured["fundamentals"]["as_of"] == "2025-09-05"

    # --- Filings
    from agents import filings as fl

    def _stub_chroma(ticker, question, *, k, item_filter, candidate_pool, use_keyword=True, as_of=None):
        captured.setdefault("filings", []).append({"as_of": as_of})
        return []

    monkeypatch.setattr(fl, "chroma_query", _stub_chroma)

    asyncio.run(fl.run({"ticker": "INTC", "thesis": {}, "as_of_date": "2025-09-05"}))
    assert captured["filings"], "filings.run did not call chroma_query"
    assert all(c["as_of"] == "2025-09-05" for c in captured["filings"])

    # --- News
    from agents import news as nw

    def _stub_tavily(ticker, company_name=None, *, days=None, max_results=None, as_of=None):
        captured["news"] = {"ticker": ticker, "as_of": as_of}
        return []

    monkeypatch.setattr(nw, "search_news", _stub_tavily)
    monkeypatch.setattr(nw, "get_financials", _stub_get_fin)

    asyncio.run(nw.run({"ticker": "INTC", "thesis": {}, "as_of_date": "2025-09-05"}))
    assert captured["news"]["as_of"] == "2025-09-05"


def test_agents_production_path_unchanged_when_no_as_of(monkeypatch):
    """When `as_of_date` is None (or absent), the data-layer calls receive
    `as_of=None`. Production behaviour preserved."""
    import asyncio

    captured: dict[str, dict] = {}
    from agents import fundamentals as fa

    def _stub_get_fin(ticker, *, as_of=None):
        captured["fundamentals"] = {"as_of": as_of}
        return {"price_history_5y": {}, "income_stmt": {}, "balance_sheet": {},
                "cash_flow": {}, "info": {"longName": "Intel Corp"}}

    monkeypatch.setattr(fa, "get_financials", _stub_get_fin)
    monkeypatch.setattr(fa, "compute_kpis", lambda f: {})

    asyncio.run(fa.run({"ticker": "INTC", "thesis": {}}))
    assert captured["fundamentals"]["as_of"] is None


# --- as-of context block injection -----------------------------------------


def test_maybe_inject_as_of_returns_unchanged_when_none():
    from utils.as_of import maybe_inject_as_of

    out = maybe_inject_as_of("ORIGINAL PROMPT", None)
    assert out == "ORIGINAL PROMPT"


def test_maybe_inject_as_of_prepends_block_when_set():
    from utils.as_of import maybe_inject_as_of

    out = maybe_inject_as_of("ORIGINAL PROMPT", "2025-09-05")
    assert out.startswith("# BACKTEST MODE")
    assert "2025-09-05" in out
    assert "ORIGINAL PROMPT" in out


def test_render_as_of_block_substitutes_date():
    from utils.as_of import render_as_of_block

    block = render_as_of_block("2025-09-05")
    assert "2025-09-05" in block
    assert "{as_of_date}" not in block, "template variable must be substituted"


# --- Monte Carlo node uses historical treasury yield -----------------------


def test_monte_carlo_passes_as_of_to_treasury(monkeypatch):
    """The MC node passes `state.as_of_date` to `get_10y_treasury_yield`
    so the discount rate reflects the historical 10y yield."""
    import asyncio

    captured: dict = {}

    from agents import __init__ as agents_init  # noqa: F401
    from data import treasury

    def _stub_yield(*, as_of=None):
        captured["as_of"] = as_of
        return 0.041

    monkeypatch.setattr(treasury, "get_10y_treasury_yield", _stub_yield)

    # Run the MC node directly — it'll bail on missing inputs, but we only
    # care that get_10y_treasury_yield got called (or not) before bail.
    # Provide enough state for the MC node to reach the yield call.
    from agents import monte_carlo
    state = {
        "ticker": "INTC",
        "thesis": {
            "name": "x", "summary": "y", "anchor_tickers": ["INTC"], "universe": ["INTC"],
            "valuation": {
                "equity_risk_premium": 0.05, "erp_basis": "x",
                "terminal_growth_rate": 0.025, "terminal_growth_basis": "x",
                "discount_rate_floor": 0.06, "discount_rate_cap": 0.15,
            },
        },
        "fundamentals": {
            "kpis": {"revenue_latest": 1e10, "shares_outstanding": 1e9, "current_price": 100.0},
            "projections": {
                "revenue_growth_mean": 0.05, "revenue_growth_std": 0.02,
                "margin_mean": 0.10, "margin_std": 0.03,
                "exit_multiple_mean": 12.0, "exit_multiple_std": 2.0,
            },
        },
        "as_of_date": "2025-09-05",
    }
    asyncio.run(monte_carlo(state))
    assert captured.get("as_of") == "2025-09-05"
