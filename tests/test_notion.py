"""Tier 1 unit tests for data/notion.py — Notion memory wrapper.

All tests run without `notion-client` actually contacting Notion. We mock
the `Client` at module level so the test suite doesn't pay API quota or
require network access. Tier 3 (real Notion writes) belongs in a future
`test_notion_integration.py` gated behind `pytest -m integration`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data import notion as nm


@pytest.fixture(autouse=True)
def _reset_data_source_cache(monkeypatch):
    """The Notion module memoises database→data_source lookups across calls
    so production code makes one retrieve per DB. Tests must start with a
    clean cache so monkeypatched stubs are exercised on every test."""
    monkeypatch.setattr(nm, "_data_source_cache", {}, raising=False)


# --- Env-var gating --------------------------------------------------------


def test_is_configured_default_false(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    assert nm.is_configured() is False


def test_is_configured_true_when_set(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "ntn_some_token")
    assert nm.is_configured() is True


def test_get_client_returns_none_when_unconfigured(monkeypatch):
    """No env var → no client → every public call is a no-op."""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    assert nm._get_client() is None


# --- write_report ----------------------------------------------------------


def _stub_client_capturing_calls() -> tuple[SimpleNamespace, dict]:
    """Build a minimal stub of the notion-client interface and a dict that
    records every call made through it. Lets us assert the write_report
    path issues the right shape of API call without paying for one."""
    captured: dict = {"pages_create": [], "blocks_append": []}

    class _Pages:
        def create(self, **kwargs):
            captured["pages_create"].append(kwargs)
            return {"id": "fake-page-id-123", "url": "https://notion.so/fake"}

        def update(self, **kwargs):
            captured.setdefault("pages_update", []).append(kwargs)
            return {"id": kwargs.get("page_id")}

        def retrieve(self, page_id, **kwargs):
            return {"id": page_id, "properties": {}}

    class _Blocks:
        class _Children:
            def append(self, **kwargs):
                captured["blocks_append"].append(kwargs)

            def list(self, **kwargs):
                return {"results": []}

        def __init__(self):
            self.children = _Blocks._Children()

    class _Databases:
        def create(self, **kwargs):
            captured.setdefault("databases_create", []).append(kwargs)
            return {"id": "fake-db-id-zzz"}

        def retrieve(self, database_id, **kwargs):
            captured.setdefault("databases_retrieve", []).append(database_id)
            return {
                "id": database_id,
                "data_sources": [
                    {"id": f"ds-of-{database_id}", "name": "default"}
                ],
            }

    class _DataSources:
        def query(self, **kwargs):
            captured.setdefault("data_sources_query", []).append(kwargs)
            return {"results": []}

    client = SimpleNamespace(
        pages=_Pages(),
        blocks=_Blocks(),
        databases=_Databases(),
        data_sources=_DataSources(),
    )
    return client, captured


def test_write_report_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    out = nm.write_report(
        ticker="NVDA",
        thesis_name="AI cake",
        markdown="# x",
    )
    assert out is None


def test_write_report_creates_page_with_properties_when_configured(monkeypatch):
    """Happy path: env vars set, valid markdown → pages.create called with
    parent.database_id, title property, ticker/thesis/date metadata, and
    children = the converted block list."""
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_REPORTS", "fake-db-id")
    monkeypatch.setattr(nm, "_client_cache", None)
    client, captured = _stub_client_capturing_calls()
    monkeypatch.setattr(nm, "_get_client", lambda: client)

    md = (
        "# NVDA — AI cake\n\n"
        "**Date:** 2026-04-30 · **Confidence:** medium\n\n"
        "## Bull case\n- Strong revenue growth\n\n"
        "## Top risks\n1. Supply concentration — severity 4\n"
    )
    url = nm.write_report(
        ticker="NVDA",
        thesis_name="AI cake",
        markdown=md,
        confidence="medium",
        p50=185.0,
        current_price=200.0,
        run_id="abc-123",
    )

    assert url == "https://notion.so/fake"
    assert len(captured["pages_create"]) == 1
    call = captured["pages_create"][0]
    # parent.database_id is set
    assert call["parent"]["database_id"] == "fake-db-id"
    # properties carry the structured metadata
    props = call["properties"]
    assert "Name" in props and "title" in props["Name"]
    assert props["Ticker"]["rich_text"][0]["text"]["content"] == "NVDA"
    assert props["Thesis"]["rich_text"][0]["text"]["content"] == "AI cake"
    assert props["Confidence"]["select"]["name"] == "medium"
    assert props["DCF P50"]["number"] == 185.0
    assert props["Current Price"]["number"] == 200.0
    assert props["Run ID"]["rich_text"][0]["text"]["content"] == "abc-123"
    # children include heading_1 + heading_2 + bullets + numbered
    block_types = [b["type"] for b in call["children"]]
    assert "heading_1" in block_types
    assert "heading_2" in block_types
    assert "bulleted_list_item" in block_types
    assert "numbered_list_item" in block_types


def test_write_report_returns_none_on_api_failure(monkeypatch):
    """notion-client raise → log + return None, NEVER propagate to caller."""
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_REPORTS", "fake-db-id")
    monkeypatch.setattr(nm, "_client_cache", None)

    class _BoomClient:
        def __init__(self):
            self.pages = SimpleNamespace(
                create=lambda **kw: (_ for _ in ()).throw(RuntimeError("notion 502"))
            )

    monkeypatch.setattr(nm, "_get_client", lambda: _BoomClient())
    out = nm.write_report(ticker="X", thesis_name="t", markdown="# x")
    assert out is None


def test_write_report_skipped_when_db_id_missing(monkeypatch):
    """Env var set but no NOTION_DB_REPORTS → no-op."""
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.delenv("NOTION_DB_REPORTS", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    client, captured = _stub_client_capturing_calls()
    monkeypatch.setattr(nm, "_get_client", lambda: client)
    out = nm.write_report(ticker="X", thesis_name="t", markdown="# x")
    assert out is None
    assert captured["pages_create"] == []


# --- read_thesis_notes -----------------------------------------------------


def test_read_thesis_notes_returns_empty_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    assert nm.read_thesis_notes("ai_cake") == ""


def test_read_thesis_notes_returns_empty_when_slug_not_found(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_THESES", "fake-db")
    monkeypatch.setattr(nm, "_client_cache", None)
    client, _ = _stub_client_capturing_calls()
    monkeypatch.setattr(nm, "_get_client", lambda: client)
    # databases.query returns empty results
    assert nm.read_thesis_notes("ai_cake") == ""


def test_read_thesis_notes_concatenates_block_text(monkeypatch):
    """When the DB has a row + blocks under it, concatenate the block text."""
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_THESES", "fake-db")
    monkeypatch.setattr(nm, "_client_cache", None)

    class _DatabasesStub:
        def retrieve(self, database_id, **kwargs):
            return {
                "id": database_id,
                "data_sources": [{"id": "ds-fake", "name": "default"}],
            }

    class _DataSourcesStub:
        def query(self, **kwargs):
            return {"results": [{"id": "page-1"}]}

    class _ChildrenStub:
        def list(self, **kwargs):
            return {
                "results": [
                    {
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"plain_text": "Trim 20% if Q3 misses $42B."}
                            ]
                        },
                    },
                    {
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"plain_text": "Watch hyperscaler capex."}
                            ]
                        },
                    },
                ]
            }

    blocks_stub = SimpleNamespace(children=_ChildrenStub())
    client = SimpleNamespace(
        databases=_DatabasesStub(),
        data_sources=_DataSourcesStub(),
        blocks=blocks_stub,
    )
    monkeypatch.setattr(nm, "_get_client", lambda: client)

    notes = nm.read_thesis_notes("ai_cake")
    assert "Trim 20%" in notes
    assert "hyperscaler capex" in notes


# --- write_alert / update_alert_status -------------------------------------


def test_write_alert_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    out = nm.write_alert(ticker="NVDA", thesis_name="AI cake", severity=4, signal="x")
    assert out is None


def test_write_alert_creates_page_with_pending_status(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_ALERTS", "alerts-db")
    monkeypatch.setattr(nm, "_client_cache", None)
    client, captured = _stub_client_capturing_calls()
    monkeypatch.setattr(nm, "_get_client", lambda: client)
    out = nm.write_alert(
        ticker="NVDA",
        thesis_name="AI cake",
        severity=4,
        signal="supply concentration",
        evidence_url="https://sec.gov/...",
        run_id="r-1",
    )
    assert out is not None
    alert_id, url = out
    assert alert_id == "fake-page-id-123"
    assert len(captured["pages_create"]) == 1
    props = captured["pages_create"][0]["properties"]
    assert props["Status"]["select"]["name"] == "pending"
    assert props["Severity"]["number"] == 4
    assert props["Evidence URL"]["url"] == "https://sec.gov/..."


def test_update_alert_status_rejects_unknown_status():
    with pytest.raises(ValueError, match="invalid status"):
        nm.update_alert_status("page-id", "approved")


def test_update_alert_status_returns_false_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    assert nm.update_alert_status("page-id", "dismissed") is False


def test_update_alert_status_calls_pages_update(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setattr(nm, "_client_cache", None)
    client, captured = _stub_client_capturing_calls()
    monkeypatch.setattr(nm, "_get_client", lambda: client)
    assert nm.update_alert_status("page-1", "actioned") is True
    update_call = captured["pages_update"][0]
    assert update_call["page_id"] == "page-1"
    assert update_call["properties"]["Status"]["select"]["name"] == "actioned"


# --- Watchlist -------------------------------------------------------------


def test_write_watchlist_items_skips_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    n = nm.write_watchlist_items(items=["x (filings)"], ticker="N", thesis_name="t")
    assert n == 0


def test_write_watchlist_items_extracts_agent_from_suffix(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_WATCHLIST", "wl-db")
    monkeypatch.setattr(nm, "_client_cache", None)
    client, captured = _stub_client_capturing_calls()
    monkeypatch.setattr(nm, "_get_client", lambda: client)
    n = nm.write_watchlist_items(
        items=[
            "Q3 earnings call (news)",
            "TSM yield (filings)",
            "FCF margin trend (fundamentals)",
        ],
        ticker="NVDA",
        thesis_name="AI cake",
        run_id="run-1",
    )
    assert n == 3
    pages = captured["pages_create"]
    assert len(pages) == 3
    agents = [p["properties"]["Agent"]["select"]["name"] for p in pages]
    assert agents == ["news", "filings", "fundamentals"]


def test_write_watchlist_items_dedupes_against_existing(monkeypatch):
    """When the Watchlist DB already has a row with the same Item, skip it."""
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_WATCHLIST", "wl-db")
    monkeypatch.setattr(nm, "_client_cache", None)
    captured = {"pages_create": []}

    class _Pages:
        def create(self, **kwargs):
            captured["pages_create"].append(kwargs)
            return {"id": "p", "url": "u"}

    class _Databases:
        def retrieve(self, database_id, **kwargs):
            return {
                "id": database_id,
                "data_sources": [{"id": "ds-fake", "name": "default"}],
            }

    class _DataSources:
        def query(self, **kwargs):
            # Pretend "Q3 earnings call (news)" already exists
            return {
                "results": [
                    {
                        "properties": {
                            "Item": {
                                "title": [
                                    {"plain_text": "Q3 earnings call (news)"}
                                ]
                            }
                        }
                    }
                ]
            }

    client = SimpleNamespace(
        pages=_Pages(), databases=_Databases(), data_sources=_DataSources()
    )
    monkeypatch.setattr(nm, "_get_client", lambda: client)

    n = nm.write_watchlist_items(
        items=[
            "Q3 earnings call (news)",  # duplicate, skipped
            "TSM yield (filings)",  # new, inserted
        ],
        ticker="NVDA",
        thesis_name="AI cake",
    )
    assert n == 1
    assert len(captured["pages_create"]) == 1
    assert (
        captured["pages_create"][0]["properties"]["Item"]["title"][0]["text"]["content"]
        == "TSM yield (filings)"
    )


# --- Markdown → blocks helpers --------------------------------------------


def test_markdown_to_blocks_handles_full_synthesis_template():
    md = (
        "# NVDA — AI cake\n\n"
        "**Date:** 2026-04-30 · **Confidence:** medium\n\n"
        "## What this means\n"
        "Plain-English summary of the report.\n\n"
        "## Bull case\n"
        "- First bullet (Fund kpis)\n"
        "- Second bullet (Filings 10-K)\n\n"
        "## Top risks\n"
        "1. First risk — severity 4 — explanation.\n"
        "2. Second risk — severity 3 — another.\n"
    )
    blocks = nm._markdown_to_blocks(md)
    block_types = [b["type"] for b in blocks]
    assert block_types[0] == "heading_1"
    assert "quote" in block_types  # date/confidence subtitle
    assert block_types.count("heading_2") == 3
    assert block_types.count("bulleted_list_item") == 2
    assert block_types.count("numbered_list_item") == 2


def test_markdown_bold_inline_renders_as_bold_annotation():
    blocks = nm._markdown_to_blocks(
        "## h\n\nText with **bold word** inside paragraph.\n"
    )
    para = next(b for b in blocks if b["type"] == "paragraph")
    rich = para["paragraph"]["rich_text"]
    bold_segments = [s for s in rich if s.get("annotations", {}).get("bold")]
    assert any(seg["text"]["content"] == "bold word" for seg in bold_segments)


def test_rich_text_handles_empty_string():
    """Empty input shouldn't crash — return a single empty segment."""
    out = nm._rich_text("")
    assert isinstance(out, list)
    assert len(out) == 1


# --- read_recent_reports ---------------------------------------------------


def test_read_recent_reports_returns_empty_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    monkeypatch.setattr(nm, "_client_cache", None)
    assert nm.read_recent_reports() == []


def test_read_recent_reports_extracts_fields_from_pages(monkeypatch):
    monkeypatch.setenv("NOTION_API_KEY", "ntn_x")
    monkeypatch.setenv("NOTION_DB_REPORTS", "reports-db")
    monkeypatch.setattr(nm, "_client_cache", None)

    class _Databases:
        def retrieve(self, database_id, **kwargs):
            return {
                "id": database_id,
                "data_sources": [{"id": "ds-fake", "name": "default"}],
            }

    class _DataSources:
        def query(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://notion.so/page1",
                        "properties": {
                            "Name": {
                                "title": [{"plain_text": "NVDA — AI cake — 2026-04-30"}]
                            },
                            "Ticker": {"rich_text": [{"plain_text": "NVDA"}]},
                            "Thesis": {"rich_text": [{"plain_text": "AI cake"}]},
                            "Confidence": {"select": {"name": "medium"}},
                            "Date": {"date": {"start": "2026-04-30"}},
                        },
                    }
                ]
            }

    monkeypatch.setattr(
        nm,
        "_get_client",
        lambda: SimpleNamespace(databases=_Databases(), data_sources=_DataSources()),
    )
    rows = nm.read_recent_reports(limit=5)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["thesis"] == "AI cake"
    assert rows[0]["confidence"] == "medium"
    assert rows[0]["url"] == "https://notion.so/page1"


# --- 2025-09-03 API regression guard ----------------------------------------


def test_notion_module_does_not_call_legacy_databases_query():
    """notion-client 3.x deprecated `databases.query`; reads must go through
    `data_sources.query`. A blanket source-text check is the cheapest way to
    keep us from regressing — if a future edit reintroduces the old call, this
    test fails before any drill-in tries to hit Notion in anger."""
    import inspect

    src = inspect.getsource(nm)
    # Strip comment-only lines so the explanatory header in `_resolve_data_source_id`
    # doesn't trip the guard. Only call-site lines should be inspected.
    code_lines = [
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert ".databases.query(" not in code_only, (
        "data/notion.py is using the deprecated `databases.query`; "
        "switch to `client.data_sources.query(data_source_id=..., ...)` and "
        "resolve the data_source_id via `_resolve_data_source_id(db_id)`."
    )
