"""Run Inspector → redirect to Mission Control's Runs tab.

Mission Control v2 (Phase 2) absorbed the Run Inspector into the Runs tab of
`ui/pages/mission_control.py`. This stub keeps the page's old address alive —
Telegram deep links and bookmarks built from `STREAMLIT_PUBLIC_URL` as
`/run_inspector?run_id=<uuid>` still land on that run: the id is carried over
both as the `run_id` query param and as the `inspect_run_id` session key the
Runs tab preselects from.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="FINAQ — Run Inspector", page_icon="🔍", layout="wide")

run_id = st.query_params.get("run_id") or st.session_state.get("inspect_run_id")
if run_id:
    st.session_state["inspect_run_id"] = run_id
    st.session_state["mc_jump_tab"] = "Runs"
st.switch_page("pages/mission_control.py", query_params={"run_id": run_id} if run_id else None)
