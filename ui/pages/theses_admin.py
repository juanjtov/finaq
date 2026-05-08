"""Theses Admin — promote / demote / archive curated & ad-hoc theses.

Streamlit twin of the Telegram `/promote` and `/demote` commands. Both
paths drive `data.theses.{promote_thesis, demote_thesis, archive_thesis}`
under the hood, so behaviour is identical regardless of entry point.

Lifecycle the page exposes:
  - Curated thesis row → Demote button → moves JSON to `theses/archive/`.
  - Ad-hoc thesis row → Promote button → strips `adhoc_` prefix and
    promotes; or Archive button → archives without rename.
  - Archive section → read-only list of every previously archived thesis
    with timestamps, so the user can spot-check that demote did its job
    and recover by hand if needed (no Restore button — recovery is rare
    and a `mv` is fine for that edge case).

The CIO heartbeat (Step 11.10) sweeps only curated theses; ad-hoc
theses are inert until promoted. So the admin page is the user's main
surface for graduating an ad-hoc into ongoing monitoring.
"""

from __future__ import annotations

# Bootstrap so this page can be loaded standalone via `streamlit run`.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import pandas as pd
import streamlit as st

from data import theses as theses_lifecycle
from ui.components import page_header, section_divider
from utils.schemas import Thesis

st.set_page_config(page_title="FINAQ — Theses Admin", page_icon="🗂️", layout="wide")

THESES_DIR = Path(__file__).parents[2] / "theses"
ARCHIVE_DIR = THESES_DIR / "archive"
ADHOC_PREFIX = "adhoc_"


# --- Helpers --------------------------------------------------------------


def _list_active_slugs() -> list[str]:
    """Slugs of every active (non-archived) thesis JSON in /theses/."""
    if not THESES_DIR.exists():
        return []
    return sorted(p.stem for p in THESES_DIR.glob("*.json"))


def _list_archive_files() -> list[Path]:
    """Archive entries sorted newest-first by filename (which embeds the ts)."""
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(ARCHIVE_DIR.glob("*.json"), reverse=True)


def _load_thesis(slug: str) -> tuple[dict | None, str | None]:
    """Read + parse `theses/{slug}.json`. Returns (data, error_msg)."""
    path = THESES_DIR / f"{slug}.json"
    if not path.exists():
        return None, f"{path} not found"
    try:
        return json.loads(path.read_text()), None
    except Exception as e:
        return None, f"parse error: {e}"


def _validates(slug: str) -> tuple[bool, str | None]:
    """Run the Pydantic schema check used by `promote_thesis`. Lets the
    page warn the user before they tap a button that would otherwise
    fail with a cryptic error."""
    path = THESES_DIR / f"{slug}.json"
    try:
        Thesis.model_validate_json(path.read_text())
        return True, None
    except Exception as e:
        return False, str(e)


# --- Section renderers ----------------------------------------------------


def _row_meta(data: dict | None) -> tuple[str, str, int]:
    """Pull (name, anchors_str, universe_size) from a thesis dict."""
    if not data:
        return "—", "—", 0
    name = str(data.get("name") or "—")
    anchors = data.get("anchor_tickers") or []
    universe = data.get("universe") or []
    anchors_str = ", ".join(anchors[:3]) + ("…" if len(anchors) > 3 else "") if anchors else "—"
    return name, anchors_str, len(universe)


def _render_curated(slugs: list[str]) -> None:
    if not slugs:
        st.caption("No curated theses. Promote an ad-hoc thesis below to get started.")
        return
    st.markdown(
        "These theses are **swept by the CIO heartbeat** (2× daily) and "
        "appear on the dashboard sidebar."
    )
    for slug in slugs:
        data, err = _load_thesis(slug)
        name, anchors_str, n = _row_meta(data)
        cols = st.columns([3, 4, 2, 1.4, 1.4])
        cols[0].markdown(f"**`{slug}`**")
        cols[1].markdown(f"{name}  \n<span style='opacity:0.7;'>anchors: {anchors_str}</span>",
                         unsafe_allow_html=True)
        cols[2].metric("Universe", n, label_visibility="visible")
        with cols[3]:
            if st.button("View JSON", key=f"view_curated_{slug}"):
                st.session_state[f"_show_json_{slug}"] = True
        with cols[4]:
            if st.button("Demote", key=f"demote_{slug}", type="secondary"):
                ok, msg = theses_lifecycle.demote_thesis(slug)
                if ok:
                    st.success(f"Demoted `{slug}` → archive")
                else:
                    st.error(f"Demote failed: {msg}")
                st.rerun()
        if err:
            st.error(f"`{slug}` JSON error: {err}")
        if st.session_state.get(f"_show_json_{slug}"):
            with st.expander(f"`{slug}.json` — full content", expanded=True):
                st.json(data or {})
                if st.button("Hide", key=f"hide_curated_{slug}"):
                    st.session_state[f"_show_json_{slug}"] = False
                    st.rerun()


def _render_adhoc(slugs: list[str]) -> None:
    if not slugs:
        st.caption(
            "No ad-hoc theses on disk. Use `/analyze TOPIC` in Telegram or "
            "the dashboard's New Thesis tab to create one."
        )
        return
    st.markdown(
        "Generated from `/analyze TOPIC` — **not yet swept by the CIO**. "
        "Promote one to graduate it into ongoing monitoring."
    )
    for slug in slugs:
        data, err = _load_thesis(slug)
        name, anchors_str, n = _row_meta(data)
        cols = st.columns([3, 4, 2, 1.4, 1.4, 1.4])
        cols[0].markdown(f"**`{slug}`**")
        cols[1].markdown(f"{name}  \n<span style='opacity:0.7;'>anchors: {anchors_str}</span>",
                         unsafe_allow_html=True)
        cols[2].metric("Universe", n, label_visibility="visible")

        with cols[3]:
            if st.button("View JSON", key=f"view_adhoc_{slug}"):
                st.session_state[f"_show_json_{slug}"] = True

        with cols[4]:
            valid, vmsg = _validates(slug)
            if st.button(
                "Promote",
                key=f"promote_{slug}",
                type="primary",
                disabled=not valid,
                help=None if valid else f"Schema invalid: {vmsg}",
            ):
                ok, msg = theses_lifecycle.promote_thesis(slug)
                if ok:
                    st.success(f"Promoted `{slug}` → curated")
                else:
                    st.error(f"Promote failed: {msg}")
                st.rerun()

        with cols[5]:
            if st.button("Archive", key=f"archive_{slug}", type="secondary"):
                ok, msg = theses_lifecycle.archive_thesis(slug)
                if ok:
                    st.success(f"Archived `{slug}`")
                else:
                    st.error(f"Archive failed: {msg}")
                st.rerun()

        if err:
            st.error(f"`{slug}` JSON error: {err}")
        if st.session_state.get(f"_show_json_{slug}"):
            with st.expander(f"`{slug}.json` — full content", expanded=True):
                st.json(data or {})
                if st.button("Hide", key=f"hide_adhoc_{slug}"):
                    st.session_state[f"_show_json_{slug}"] = False
                    st.rerun()


def _render_archive(files: list[Path]) -> None:
    if not files:
        st.caption("Archive is empty.")
        return
    st.caption(
        f"{len(files)} archived thesis file(s). Sorted newest first by filename. "
        "To restore: rename the file back to `theses/{slug}.json` manually."
    )
    rows: list[dict] = []
    for p in files:
        # Filename format: `YYYYMMDD_HHMMSS__slug.json` (or with `_N` suffix)
        try:
            ts_part, rest = p.stem.split("__", 1)
            archived_at = (
                f"{ts_part[:4]}-{ts_part[4:6]}-{ts_part[6:8]} "
                f"{ts_part[9:11]}:{ts_part[11:13]}:{ts_part[13:15]}"
            )
        except ValueError:
            archived_at = "—"
            rest = p.stem
        rows.append({"slug": rest, "archived_at_utc": archived_at, "filename": p.name})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch")


# --- Page body ------------------------------------------------------------


def main() -> None:
    page_header(
        "Theses Admin",
        "Promote ad-hoc theses to curated, demote curated to archive.",
    )

    active = _list_active_slugs()
    curated = [s for s in active if not s.startswith(ADHOC_PREFIX)]
    adhoc = [s for s in active if s.startswith(ADHOC_PREFIX)]
    archive_files = _list_archive_files()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Curated", len(curated))
    metric_cols[1].metric("Ad-hoc", len(adhoc))
    metric_cols[2].metric("Archived", len(archive_files))

    section_divider()
    st.subheader("Curated theses")
    _render_curated(curated)

    section_divider()
    st.subheader("Ad-hoc theses")
    _render_adhoc(adhoc)

    section_divider()
    st.subheader("Archive")
    _render_archive(archive_files)


main()
