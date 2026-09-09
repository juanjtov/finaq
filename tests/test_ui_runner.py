"""Tests for ui/_runner.py — the background drill-in registry the dashboard
polls so a run survives Streamlit page navigation."""

from __future__ import annotations

import threading

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
