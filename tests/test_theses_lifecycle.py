"""Tests for `data/theses.py` — promote / demote / archive lifecycle.

Each test uses an isolated `tmp_path/theses/` so we never touch the
curated theses checked into the repo.
"""

from __future__ import annotations

import json

import pytest

from data import theses as theses_lifecycle


@pytest.fixture
def isolated_theses_dir(tmp_path, monkeypatch):
    """Point THESES_DIR + ARCHIVE_DIR at a fresh tmp_path so tests don't
    touch the real `theses/` checked into the repo."""
    fake_theses = tmp_path / "theses"
    fake_archive = fake_theses / "archive"
    fake_theses.mkdir()
    fake_archive.mkdir()
    monkeypatch.setattr(theses_lifecycle, "THESES_DIR", fake_theses)
    monkeypatch.setattr(theses_lifecycle, "ARCHIVE_DIR", fake_archive)
    return fake_theses


def _valid_thesis_json(name: str = "Test Thesis") -> str:
    """Minimal Thesis JSON that validates."""
    return json.dumps(
        {
            "name": name,
            "summary": "Short summary for the test thesis.",
            "anchor_tickers": ["AAPL"],
            "universe": ["AAPL", "MSFT"],
            "relationships": [],
            "material_thresholds": [],
        }
    )


# --- archive_thesis -------------------------------------------------------


def test_archive_thesis_happy_path(isolated_theses_dir):
    src = isolated_theses_dir / "ai_cake.json"
    src.write_text(_valid_thesis_json("AI cake"))

    ok, msg = theses_lifecycle.archive_thesis("ai_cake")

    assert ok is True
    assert "archived" in msg.lower()
    assert not src.exists()
    archived = list((isolated_theses_dir / "archive").glob("*__ai_cake.json"))
    assert len(archived) == 1


def test_archive_thesis_missing_source(isolated_theses_dir):
    ok, msg = theses_lifecycle.archive_thesis("does_not_exist")
    assert ok is False
    assert "not found" in msg


def test_archive_thesis_skips_validation(isolated_theses_dir):
    """A malformed thesis must still be archivable — that's the whole point
    of the admin-page Archive button (clean up bad files)."""
    src = isolated_theses_dir / "broken.json"
    src.write_text("{ this is not json")

    ok, msg = theses_lifecycle.archive_thesis("broken")

    assert ok is True
    assert not src.exists()


def test_archive_thesis_collision_within_same_second(isolated_theses_dir, monkeypatch):
    """Two archives of the same slug at the same timestamp must produce
    distinct files via the `_N` suffix path."""
    monkeypatch.setattr(theses_lifecycle, "_archive_timestamp", lambda: "20260430_174812")

    src = isolated_theses_dir / "ai_cake.json"
    src.write_text(_valid_thesis_json("v1"))
    ok1, _ = theses_lifecycle.archive_thesis("ai_cake")
    assert ok1

    src.write_text(_valid_thesis_json("v2"))
    ok2, _ = theses_lifecycle.archive_thesis("ai_cake")
    assert ok2

    archive_files = sorted((isolated_theses_dir / "archive").glob("*.json"))
    assert [p.name for p in archive_files] == [
        "20260430_174812__ai_cake.json",
        "20260430_174812__ai_cake_1.json",
    ]


# --- promote_thesis -------------------------------------------------------


def test_promote_happy_path(isolated_theses_dir):
    src = isolated_theses_dir / "adhoc_defense_semis.json"
    src.write_text(_valid_thesis_json("Defense semis"))

    ok, msg = theses_lifecycle.promote_thesis("adhoc_defense_semis")

    assert ok is True
    assert "promoted" in msg.lower()
    assert (isolated_theses_dir / "defense_semis.json").exists()
    assert not src.exists()


def test_promote_overwrites_curated_via_archive(isolated_theses_dir):
    """If a curated thesis already exists at the target name, it must be
    archived (not destroyed) before the adhoc takes its place."""
    existing = isolated_theses_dir / "defense_semis.json"
    existing.write_text(_valid_thesis_json("Old defense semis"))
    src = isolated_theses_dir / "adhoc_defense_semis.json"
    src.write_text(_valid_thesis_json("New defense semis"))

    ok, _ = theses_lifecycle.promote_thesis("adhoc_defense_semis")
    assert ok

    archived = list((isolated_theses_dir / "archive").glob("*__defense_semis.json"))
    assert len(archived) == 1
    new_curated = json.loads((isolated_theses_dir / "defense_semis.json").read_text())
    assert new_curated["name"] == "New defense semis"
    archived_data = json.loads(archived[0].read_text())
    assert archived_data["name"] == "Old defense semis"


def test_promote_rejects_non_adhoc_slug(isolated_theses_dir):
    src = isolated_theses_dir / "ai_cake.json"
    src.write_text(_valid_thesis_json())

    ok, msg = theses_lifecycle.promote_thesis("ai_cake")

    assert ok is False
    assert "adhoc" in msg.lower()
    assert src.exists()


def test_promote_missing_source(isolated_theses_dir):
    ok, msg = theses_lifecycle.promote_thesis("adhoc_does_not_exist")
    assert ok is False
    assert "not found" in msg


def test_promote_rejects_invalid_schema(isolated_theses_dir):
    """A malformed adhoc thesis must NOT enter the curated set."""
    src = isolated_theses_dir / "adhoc_garbage.json"
    src.write_text(json.dumps({"name": "x"}))  # missing required fields

    ok, msg = theses_lifecycle.promote_thesis("adhoc_garbage")

    assert ok is False
    assert "schema" in msg.lower() or "validation" in msg.lower()
    assert src.exists()  # source untouched on validation failure
    assert not (isolated_theses_dir / "garbage.json").exists()


def test_promote_rejects_unparseable_json(isolated_theses_dir):
    src = isolated_theses_dir / "adhoc_garbage.json"
    src.write_text("{ not json {{{")

    ok, msg = theses_lifecycle.promote_thesis("adhoc_garbage")

    assert ok is False
    assert "schema" in msg.lower() or "validation" in msg.lower()
    assert src.exists()


def test_promote_rejects_empty_curated_name(isolated_theses_dir):
    """`adhoc_` alone (prefix-only, no body) must refuse — would promote
    to an empty curated filename."""
    src = isolated_theses_dir / "adhoc_.json"
    src.write_text(_valid_thesis_json())

    ok, msg = theses_lifecycle.promote_thesis("adhoc_")

    assert ok is False
    assert "empty" in msg.lower()


# --- demote_thesis --------------------------------------------------------


def test_demote_happy_path(isolated_theses_dir):
    src = isolated_theses_dir / "ai_cake.json"
    src.write_text(_valid_thesis_json())

    ok, msg = theses_lifecycle.demote_thesis("ai_cake")

    assert ok is True
    assert "archived" in msg.lower()
    assert not src.exists()
    archived = list((isolated_theses_dir / "archive").glob("*__ai_cake.json"))
    assert len(archived) == 1


def test_demote_rejects_adhoc_slug(isolated_theses_dir):
    src = isolated_theses_dir / "adhoc_defense_semis.json"
    src.write_text(_valid_thesis_json())

    ok, msg = theses_lifecycle.demote_thesis("adhoc_defense_semis")

    assert ok is False
    assert "adhoc" in msg.lower()
    assert src.exists()


def test_demote_missing_source(isolated_theses_dir):
    ok, msg = theses_lifecycle.demote_thesis("does_not_exist")
    assert ok is False
    assert "not found" in msg


def test_demote_idempotent_after_first_call(isolated_theses_dir):
    """Calling demote twice on the same slug: second call returns
    (False, not found) — the file is gone after the first move."""
    src = isolated_theses_dir / "ai_cake.json"
    src.write_text(_valid_thesis_json())

    ok1, _ = theses_lifecycle.demote_thesis("ai_cake")
    assert ok1

    ok2, msg = theses_lifecycle.demote_thesis("ai_cake")
    assert ok2 is False
    assert "not found" in msg


# --- Review age (2026-09-07) ----------------------------------------------


def _write(dir_, slug: str, **extra) -> None:
    data = json.loads(_valid_thesis_json(slug))
    data.update(extra)
    (dir_ / f"{slug}.json").write_text(json.dumps(data))


def test_review_age_days_reads_last_reviewed(isolated_theses_dir):
    from datetime import date

    _write(isolated_theses_dir, "alpha", last_reviewed="2026-06-01")
    age = theses_lifecycle.review_age_days("alpha", today=date(2026, 9, 7))
    assert age == 98


def test_review_age_days_falls_back_to_mtime_when_field_absent(isolated_theses_dir):
    import os
    import time
    from datetime import date

    _write(isolated_theses_dir, "beta")  # no last_reviewed
    path = isolated_theses_dir / "beta.json"
    ten_days_ago = time.time() - 10 * 86400
    os.utime(path, (ten_days_ago, ten_days_ago))
    age = theses_lifecycle.review_age_days("beta", today=date.today())
    assert age in (10, 11)  # UTC date rounding at the boundary


def test_review_age_days_malformed_field_falls_back_to_mtime(isolated_theses_dir):
    _write(isolated_theses_dir, "gamma", last_reviewed="not-a-date")
    assert theses_lifecycle.review_age_days("gamma") == 0  # just written


def test_review_age_days_none_for_missing_file(isolated_theses_dir):
    assert theses_lifecycle.review_age_days("nope") is None


def test_mark_reviewed_stamps_today_and_preserves_other_keys(isolated_theses_dir):
    from datetime import date

    _write(isolated_theses_dir, "delta", last_reviewed="2026-01-01")
    ok, msg = theses_lifecycle.mark_reviewed("delta", today=date(2026, 9, 7))
    assert ok and "2026-09-07" in msg
    data = json.loads((isolated_theses_dir / "delta.json").read_text())
    assert data["last_reviewed"] == "2026-09-07"
    assert data["universe"] == ["AAPL", "MSFT"]  # untouched
    assert theses_lifecycle.review_age_days("delta", today=date(2026, 9, 7)) == 0


def test_mark_reviewed_missing_file(isolated_theses_dir):
    ok, msg = theses_lifecycle.mark_reviewed("nope")
    assert not ok and "not found" in msg


def test_overdue_theses_lists_only_past_threshold_oldest_first(isolated_theses_dir):
    from datetime import date

    today = date(2026, 9, 7)
    _write(isolated_theses_dir, "old", last_reviewed="2026-04-29")      # 131d
    _write(isolated_theses_dir, "older", last_reviewed="2026-01-01")    # 249d
    _write(isolated_theses_dir, "fresh", last_reviewed="2026-08-01")    # 37d
    _write(isolated_theses_dir, "edge", last_reviewed="2026-06-09")     # 90d → not overdue
    out = theses_lifecycle.overdue_theses(
        ["old", "older", "fresh", "edge", "missing"], today=today,
    )
    assert [o["slug"] for o in out] == ["older", "old"]
    assert out[0]["age_days"] == 249
    assert out[1]["age_days"] == 131


def test_promote_stamps_last_reviewed(isolated_theses_dir):
    """Promotion is a human judgement → counts as a review."""
    from datetime import date

    (isolated_theses_dir / "adhoc_zeta.json").write_text(_valid_thesis_json("Zeta"))
    ok, _ = theses_lifecycle.promote_thesis("adhoc_zeta")
    assert ok
    data = json.loads((isolated_theses_dir / "zeta.json").read_text())
    assert data["last_reviewed"] == date.today().isoformat()
