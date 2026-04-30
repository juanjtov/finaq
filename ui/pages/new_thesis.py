"""New Thesis page — form-based thesis creator.

Lets the user create a new persistent thesis JSON without writing the file
by hand. The form mirrors the Pydantic schema in `utils/schemas.py`:
  - name + summary
  - anchor_tickers + universe
  - relationships (data editor table)
  - material_thresholds (data editor table)
  - valuation block (equity_risk_premium, terminal_growth_rate, floor, cap,
    plus required `_basis` strings)

On submit, validates via `Thesis.model_validate(...)` and writes to
`theses/<slug>.json`. Phase 1 Triage will pick up new theses on next run.

Note: this page creates *persistent* theses. The Phase 1 Telegram
`/analyze TOPIC` command (Step 10) creates *transient* AI-synthesized
theses for one-shot drill-ins; the two paths are intentionally separate.
"""

from __future__ import annotations

# Bootstrap (see ui/app.py for explanation).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import re

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from ui.components import page_header, section_divider
from utils.schemas import Thesis

st.set_page_config(page_title="FINAQ — New Thesis", page_icon="📝", layout="wide")

THESES_DIR = Path(__file__).parents[2] / "theses"


# --- Helpers ----------------------------------------------------------------


def _slugify(name: str) -> str:
    """Filesystem-safe slug. Mirrors the convention of existing thesis files
    (lowercase, underscores)."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "thesis"


def _existing_slugs() -> set[str]:
    return {p.stem for p in THESES_DIR.glob("*.json")}


# --- Form ------------------------------------------------------------------


def _form_basics() -> tuple[str, str, list[str], list[str]]:
    cols = st.columns([2, 1])
    with cols[0]:
        name = st.text_input(
            "Thesis name",
            value=st.session_state.get("nt_name", ""),
            help="Human-readable name (e.g. 'AI cake', 'Construction'). Used in report headers.",
        )
        st.session_state["nt_name"] = name
        summary = st.text_area(
            "Summary",
            value=st.session_state.get("nt_summary", ""),
            height=120,
            help="One paragraph: what does this thesis bet on, and why? "
            "This text is read by every downstream agent — make it concrete.",
        )
        st.session_state["nt_summary"] = summary
    with cols[1]:
        slug_preview = _slugify(name) if name else "(auto)"
        st.markdown("**Filename**")
        st.code(f"theses/{slug_preview}.json", language="bash")
        if name and slug_preview in _existing_slugs():
            st.warning(f"`{slug_preview}.json` already exists — submission will overwrite.")

    universe_raw = st.text_input(
        "Universe (tickers, comma-separated)",
        value=st.session_state.get("nt_universe", ""),
        help="The full list of tickers this thesis covers, e.g. `NVDA, AVGO, TSM`.",
    )
    st.session_state["nt_universe"] = universe_raw
    universe = [t.strip().upper() for t in universe_raw.split(",") if t.strip()]

    anchors_raw = st.text_input(
        "Anchor tickers (subset of universe, comma-separated)",
        value=st.session_state.get("nt_anchors", ""),
        help="The 1–3 load-bearing names. Triage prioritises alerts on these.",
    )
    st.session_state["nt_anchors"] = anchors_raw
    anchors = [t.strip().upper() for t in anchors_raw.split(",") if t.strip()]

    return name, summary, universe, anchors


def _form_relationships(universe: list[str]) -> list[dict]:
    st.markdown("### Relationships")
    st.caption(
        "Directed edges between tickers (supplier, customer, peer, competitor). "
        "All endpoints must be in the universe above."
    )
    seed = st.session_state.get(
        "nt_relationships_df",
        pd.DataFrame(
            [{"from": "", "to": "", "type": "supplier", "note": ""}]
        ),
    )
    edited = st.data_editor(
        seed,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "from": st.column_config.TextColumn("from", width="small"),
            "to": st.column_config.TextColumn("to", width="small"),
            "type": st.column_config.SelectboxColumn(
                "type",
                options=["supplier", "customer", "peer", "competitor"],
                width="small",
            ),
            "note": st.column_config.TextColumn("note", width="large"),
        },
    )
    st.session_state["nt_relationships_df"] = edited
    rows = edited.fillna("").to_dict(orient="records")
    return [r for r in rows if r.get("from") and r.get("to")]


def _form_thresholds() -> list[dict]:
    st.markdown("### Material thresholds")
    st.caption(
        "Phase 1 Triage uses these to fire alerts. operator: > | < | abs > | contains. "
        "value type must match operator: numeric for the first three, string for `contains`."
    )
    seed = st.session_state.get(
        "nt_thresholds_df",
        pd.DataFrame(
            [
                {
                    "signal": "fcf_yield",
                    "operator": "<",
                    "value": "4",
                    "unit": "percent",
                }
            ]
        ),
    )
    edited = st.data_editor(
        seed,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "signal": st.column_config.TextColumn("signal", width="medium"),
            "operator": st.column_config.SelectboxColumn(
                "operator",
                options=[">", "<", "abs >", "contains"],
                width="small",
            ),
            "value": st.column_config.TextColumn(
                "value",
                help="Numeric for >, <, abs >. String for contains.",
                width="medium",
            ),
            "unit": st.column_config.TextColumn("unit", width="small"),
        },
    )
    st.session_state["nt_thresholds_df"] = edited
    rows: list[dict] = []
    for r in edited.fillna("").to_dict(orient="records"):
        signal = (r.get("signal") or "").strip()
        if not signal:
            continue
        operator = (r.get("operator") or "").strip()
        value_raw = (r.get("value") or "").strip()
        if not value_raw:
            continue
        # Coerce numeric for non-`contains` operators
        if operator != "contains":
            try:
                value: float | str = float(value_raw)
            except ValueError:
                value = value_raw  # let Pydantic surface the error
        else:
            value = value_raw
        rows.append(
            {
                "signal": signal,
                "operator": operator,
                "value": value,
                "unit": (r.get("unit") or "").strip(),
            }
        )
    return rows


def _form_valuation() -> dict:
    st.markdown("### Valuation block")
    st.caption(
        "Per-thesis Monte Carlo parameters. The `_basis` strings are required — "
        "they document why each value was chosen and surface in the Methodology tab."
    )
    cols = st.columns(2)
    with cols[0]:
        erp = st.number_input(
            "Equity risk premium",
            min_value=0.0,
            max_value=0.20,
            step=0.005,
            value=st.session_state.get("nt_erp", 0.05),
            format="%.3f",
            help="Long-run S&P 500 ERP (~4.5%) plus a thesis-specific premium for cyclicality.",
        )
        st.session_state["nt_erp"] = erp
        terminal = st.number_input(
            "Terminal growth rate",
            min_value=0.0,
            max_value=0.05,
            step=0.005,
            value=st.session_state.get("nt_terminal", 0.025),
            format="%.3f",
            help="Long-run nominal growth (real GDP + inflation). Bounded at 5% by Pydantic.",
        )
        st.session_state["nt_terminal"] = terminal
    with cols[1]:
        floor = st.number_input(
            "Discount-rate floor",
            min_value=0.04,
            max_value=0.20,
            step=0.005,
            value=st.session_state.get("nt_floor", 0.075),
            format="%.3f",
        )
        st.session_state["nt_floor"] = floor
        cap = st.number_input(
            "Discount-rate cap",
            min_value=0.05,
            max_value=0.25,
            step=0.005,
            value=st.session_state.get("nt_cap", 0.13),
            format="%.3f",
        )
        st.session_state["nt_cap"] = cap

    erp_basis = st.text_area(
        "ERP basis (rationale)",
        value=st.session_state.get("nt_erp_basis", ""),
        height=70,
        help="Why this ERP value? Cited in the Methodology tab.",
    )
    st.session_state["nt_erp_basis"] = erp_basis

    terminal_basis = st.text_area(
        "Terminal-growth basis (rationale)",
        value=st.session_state.get("nt_terminal_basis", ""),
        height=70,
        help="Why this terminal growth rate? Cited in the Methodology tab.",
    )
    st.session_state["nt_terminal_basis"] = terminal_basis

    return {
        "equity_risk_premium": erp,
        "erp_basis": erp_basis or "User-entered.",
        "terminal_growth_rate": terminal,
        "terminal_growth_basis": terminal_basis or "User-entered.",
        "discount_rate_floor": floor,
        "discount_rate_cap": cap,
    }


# --- Main -------------------------------------------------------------------


def main() -> None:
    page_header(
        "New thesis",
        subtitle=(
            "Create a new persistent thesis. Validated via Pydantic; written to "
            "`theses/<slug>.json` on submit. Phase 1 Triage picks it up automatically."
        ),
    )

    name, summary, universe, anchors = _form_basics()

    section_divider()
    relationships = _form_relationships(universe)

    section_divider()
    thresholds = _form_thresholds()

    section_divider()
    valuation = _form_valuation()

    section_divider()

    if st.button("Validate + save thesis", type="primary"):
        if not name:
            st.error("Name is required.")
            return
        if not universe:
            st.error("Universe is required (at least one ticker).")
            return
        if not anchors:
            st.error("At least one anchor ticker is required.")
            return

        slug = _slugify(name)
        thesis_dict = {
            "name": name,
            "summary": summary,
            "anchor_tickers": anchors,
            "universe": universe,
            "relationships": relationships,
            "material_thresholds": thresholds,
            "valuation": valuation,
        }

        # Pydantic does the heavy lifting for cross-field constraints
        # (anchors ⊆ universe, relationships → universe, value-type matches operator).
        try:
            Thesis.model_validate(thesis_dict)
        except ValidationError as e:
            st.error("Validation failed:")
            st.code(str(e))
            return

        path = THESES_DIR / f"{slug}.json"
        path.write_text(json.dumps(thesis_dict, indent=2, default=str))
        st.success(f"Thesis saved to `{path.name}`.")
        st.balloons()

        # Show what was written so the user can copy/edit by hand later
        with st.expander("Saved JSON"):
            st.json(thesis_dict)


main()
