"""Tier 1 unit tests for scripts/bootstrap_notion.py.

Notion API 2025-09-03 moved the schema onto a "data source" — the
`databases.create` call must pass `initial_data_source={"properties": ...}`
rather than `properties=...` directly. This regression bit us once and
left the user's workspace with 4 schemaless databases. These tests freeze
the call shape so we don't repeat that.
"""

from __future__ import annotations

from types import SimpleNamespace

from scripts import bootstrap_notion as bn


def test_create_db_uses_initial_data_source(monkeypatch):
    """_create_db must pass `initial_data_source={"properties": ...}` and
    must NOT pass a top-level `properties` (silently dropped under
    notion-client 3.x / API 2025-09-03)."""
    captured: dict = {}

    class _Databases:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {"id": "fake-db-id"}

    client = SimpleNamespace(databases=_Databases())
    schema = {"Name": {"title": {}}, "Slug": {"rich_text": {}}}

    db_id = bn._create_db(client, "parent-123", "FINAQ — X", schema)

    assert db_id == "fake-db-id"
    assert "initial_data_source" in captured, (
        "missing initial_data_source — schema would be silently dropped"
    )
    assert captured["initial_data_source"] == {"properties": schema}
    assert "properties" not in captured, (
        "top-level properties is a no-op under API 2025-09-03 and indicates "
        "a regression — use initial_data_source instead"
    )
    # parent + title must still be passed
    assert captured["parent"] == {"type": "page_id", "page_id": "parent-123"}
    assert captured["title"][0]["text"]["content"] == "FINAQ — X"


def test_find_existing_db_filters_archived(monkeypatch):
    """`databases.retrieve` must be checked for archived/in_trash — Notion's
    blocks.children.list returns trashed child_database blocks too, and we
    don't want to return the ID of a dead DB as 'already exists'."""

    class _Blocks:
        class _Children:
            def list(self, **kwargs):
                return {
                    "results": [
                        {
                            "type": "child_database",
                            "id": "live-id",
                            "child_database": {"title": "FINAQ — X"},
                        },
                        {
                            "type": "child_database",
                            "id": "trashed-id",
                            "child_database": {"title": "FINAQ — X"},
                        },
                    ]
                }

        def __init__(self):
            self.children = _Blocks._Children()

    class _Databases:
        def retrieve(self, database_id, **kwargs):
            return {
                "id": database_id,
                "archived": database_id == "trashed-id",
                "in_trash": database_id == "trashed-id",
            }

    client = SimpleNamespace(blocks=_Blocks(), databases=_Databases())
    found = bn._find_existing_db(client, "parent-id", "FINAQ — X")
    # Order in `results` puts trashed AFTER live — but if iteration picks
    # trashed first, the filter must skip it. Either live-id or None.
    # Crucially: never trashed-id.
    assert found != "trashed-id"
    assert found == "live-id"


def test_seed_thesis_rows_uses_data_sources_query(monkeypatch, tmp_path):
    """The dedupe-query before insert must hit `data_sources.query`, not
    the deprecated `databases.query`."""
    captured: dict = {"ds_query": [], "pages_create": []}

    class _Databases:
        def retrieve(self, database_id, **kwargs):
            return {
                "id": database_id,
                "data_sources": [{"id": "ds-fake", "name": "default"}],
            }

    class _DataSources:
        def query(self, **kwargs):
            captured["ds_query"].append(kwargs)
            return {"results": []}

    class _Pages:
        def create(self, **kwargs):
            captured["pages_create"].append(kwargs)
            return {"id": "p"}

    client = SimpleNamespace(
        databases=_Databases(), data_sources=_DataSources(), pages=_Pages()
    )

    # Stand up a temp /theses/ dir with one valid JSON
    theses_dir = tmp_path / "theses"
    theses_dir.mkdir()
    (theses_dir / "ai_cake.json").write_text(
        '{"name": "AI cake", "universe": ["NVDA", "AVGO"], '
        '"anchor_tickers": ["NVDA"]}'
    )
    monkeypatch.chdir(tmp_path)

    inserted = bn._seed_thesis_rows(client, "fake-db-id")
    assert inserted == 1
    assert len(captured["ds_query"]) == 1, "should issue dedupe query per slug"
    assert captured["ds_query"][0]["data_source_id"] == "ds-fake"
    assert len(captured["pages_create"]) == 1
    assert captured["pages_create"][0]["parent"]["database_id"] == "fake-db-id"
