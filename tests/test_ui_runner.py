"""Tests for ui/_runner.py — the background drill-in registry the dashboard
polls so a run survives Streamlit page navigation."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ui import _runner


def test_active_runs_lists_only_live_threads(monkeypatch):
    gate = threading.Event()
    live = threading.Thread(target=gate.wait, daemon=True)
    live.start()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    monkeypatch.setattr(
        _runner,
        "_active_runs",
        {
            ("NU", "nu"): {"thread": live, "started_at": 0.0, "run_id": None, "error": None},
            ("NKE", "nke"): {"thread": dead, "started_at": 0.0, "run_id": None, "error": None},
            ("COUR", "cour"): {"thread": None, "started_at": 0.0, "run_id": None, "error": None},
        },
    )
    try:
        runs = _runner.active_runs()
        assert [(t, s) for t, s, _ in runs] == [("NU", "nu")]
        assert runs[0][2] > 0
        assert _runner.is_running("nu", "nu")
        assert not _runner.is_running("NKE", "nke")
    finally:
        gate.set()


# --- finish_drill: what every real drill owes after the graph returns -------


def test_finish_drill_saves_then_indexes_with_the_notion_link(tmp_path, monkeypatch):
    """Order is the contract: Notion stamps its URL on the state, the save
    writes that to disk, and the reports index is built from the saved file."""
    monkeypatch.chdir(tmp_path)
    steps: list[str] = []
    monkeypatch.setattr(_runner, "_maybe_run_live_eval", lambda f, t: steps.append("eval"))

    def _notion(final, thesis):
        steps.append("notion")
        final["notion_report_url"] = "https://notion.example/r"

    indexed: list[dict] = []

    def _index(path):
        steps.append("index")
        indexed.append(json.loads(path.read_text()))

    monkeypatch.setattr(_runner, "_maybe_write_to_notion", _notion)
    monkeypatch.setattr(_runner, "_index_report", _index)

    final = {"ticker": "nu", "run_id": "abcdef1234", "report": "# NU", "messages": ["x"]}
    path = _runner.finish_drill(final, {"name": "Nu"}, "nu")

    assert steps == ["eval", "notion", "index"]
    assert path == Path("data_cache/demos/NU__nu__abcdef12.json")
    assert indexed[0]["notion_report_url"] == "https://notion.example/r"
    assert "messages" not in indexed[0]


def test_index_report_soft_fails(tmp_path, monkeypatch):
    """An embedding / Pinecone outage must not fail a finished drill."""
    from scripts import index_existing_reports as indexer

    def _boom(path, **kw):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(indexer, "index_one", _boom)
    _runner._index_report(tmp_path / "NU__nu__abcdef12.json")  # must not raise


def test_index_one_ids_use_the_stem_and_metadata_the_real_run_id(tmp_path, monkeypatch):
    """`fetch_reports` finds a ticker's chunks by the `{TICKER}__` id prefix,
    while the planner echoes metadata `run_id` back as `reuse_run_id` — that
    one has to be a real graph_runs id, not the filename."""
    from scripts import index_existing_reports as indexer

    captured: dict = {}

    def _upsert(ids, docs, metas):
        captured.update(ids=ids, metas=metas)
        return len(ids)

    monkeypatch.setattr(indexer, "upsert_reports", _upsert)
    path = tmp_path / "NU__nu__abcdef12.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "abcdef12-0000-4000-8000-000000000000",
                "report": "# NU\n\n**Date:** 2026-11-11\n\n## Watchlist\n- Q4 print (news)\n",
            }
        )
    )

    assert indexer.index_one(path) == 1
    assert captured["ids"] == ["NU__nu__abcdef12-watchlist"]
    meta = captured["metas"][0]
    assert meta["run_id"] == "abcdef12-0000-4000-8000-000000000000"
    assert (meta["ticker"], meta["thesis"], meta["section"]) == ("NU", "nu", "Watchlist")
