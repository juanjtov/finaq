# Mission Control v2 — live execution view (plan, 2026-10-16)

**Status:** approved 2026-10-16. All framing decisions are taken (§0, §8). Phase 1 may
start; Phase 2 follows the merge of PR #8 (`feat/discovery-graph-viz`). Nothing in this
document is built yet. Interactive mockups: the "FINAQ Mission Control Mockups" artifact linked from
the session that produced this file.

## §0 Decisions already taken

| Question | Decision | Consequence |
|---|---|---|
| Where does the live view run? | **Streamlit page + embedded HTML/JS component.** A `st.fragment(run_every="1s")` polls `state.db`, builds one snapshot, and feeds it to `st.components.v1.html`. The component animates client-side between polls. | No new dependency, no second port, works unchanged on the droplet. The iframe only re-mounts when the snapshot changes. |
| Which executions? | **Drill-in graph + CIO cycles.** Discovery / ad-hoc / Telegram router / Q&A / backtest come later through the same span table. | The CIO cycle gets step-level spans; the drill-in gets node-entry rows. |
| Full prompts? | **Full text in `state.db`**, system prompts deduplicated by content hash, retention window (default 90 days) prunes blobs but keeps excerpts and telemetry forever. | One store, offline, backed up with `data_cache/`. |
| Relation to existing pages? | **One Mission Control page with Live / Runs / Health tabs.** Run Inspector is absorbed into Runs; its `?run_id=` deep link redirects. Health keeps today's panels. | `ui/pages/run_inspector.py` becomes a redirect stub; `ui/pages/mission_control.py` is restructured. |
| Which layout? | **Mockup C (control room graph) for the Live tab; Mockup B's waterfall and drawer for the Runs tab.** | One snapshot and one inspector drawer serve both views; A is not built. |
| Theme, poll, retention | **Day palette (CLAUDE.md §13), 1 s poll, `PROMPT_RETENTION_DAYS=90`.** | Night stays a token set in the component, off by default. |
| Branch order | **Merge PR #8 (`feat/discovery-graph-viz`) before Phase 2.** Phase 1 may start now. | Phase 2 rebases on main after that merge. |

## §1 What adlc Mission Control does, and what carries over

`~/developer/projects/adlc-pipeline/mission-control/` is a stdlib Python HTTP server on
`localhost:4319` that serves one vanilla-JS page. It receives Claude Code hook events and
OTLP telemetry, mirrors GitHub into SQLite, and pushes one **state document** to the page
over Server-Sent Events whenever anything changes. The page shows an **assembly line**:
one station per agent, requests as crates moving between stations, and for the selected
station or request an inspector with *what the agent is doing now, the task prompt, the
final report, tokens / cost / latency and permissions*.

Carried over to FINAQ:

- **One state document, one renderer.** The server (for us: `data/state.py`) builds a
  single JSON snapshot; the page only renders it. Replay of a past run is the same
  renderer fed a stored snapshot.
- **"What went in / what came out" per station**, with telemetry beside it.
- **Read-only.** The page never starts, stops or edits anything.
- **Demo / replay mode** so the view can be exercised without a live run.
- **Capture is explicit.** adlc only records while its server runs; FINAQ records always,
  because the writers are our own code, not hooks.

Not carried over: the separate server and port (decided against), the GitHub poller and
hook ingestion (FINAQ instruments its own code directly), the five-colour per-agent
palette (CLAUDE.md §13 keeps a single sage accent; state is encoded in form).

## §2 Telemetry audit — what FINAQ records today and the gaps

| Signal the live view needs | Today | Gap |
|---|---|---|
| Run started / finished | `graph_runs` (`start_graph_run` / `finish_graph_run`) | 4 stale `running` rows from killed processes; no sweeper, no `source` (dashboard / CIO / Telegram / backtest). |
| **Which node is running now** | `node_runs` row written **only on node exit** (`_safe_node` `finally`) | A live view cannot see a running node. Needs a row at node entry, updated at exit. |
| Per-call telemetry | `llm_calls` row written after the response (model, latency, tokens, cost) | Written after the fact; an in-flight call is invisible. |
| System + user prompt, output | `prompt_excerpt` / `response_excerpt` clipped to 2,000 chars | The system prompt alone is 3.5–13 KB, so the user prompt never survives the clip. Synthesis user prompts reach hundreds of KB. |
| CIO cycle progress | `cio_runs` (start / finish, rollups) + `cio_actions` (decision, tokens, cost, latency, `decision_json`) | No step timeline (probe → gates → decide → budget → drills → summary → notify). Planner LLM calls land in `llm_calls` with `run_id NULL` and `node ''` (982 rows today) because `cio/planner.py` binds an unnamed accumulator. |
| Concurrency across processes | `ui/_runner.py` module dict (dashboard process only) | The CIO (launchd) and the Telegram bot are separate processes; only `state.db` sees all three. |
| Safe concurrent access | `sqlite3.connect` with defaults | No WAL, no `busy_timeout`: a 1 s reader plus two writers will hit "database is locked". |
| UI building blocks | `components.html` + mermaid precedent (Architecture page), Streamlit 1.56 (`st.fragment(run_every=…)`), `ui/components.py` palette constants, Run Inspector's waterfall + call trace | Run Inspector (652 lines) and Mission Control (905 lines) overlap; neither is live. |

Side finding from the audit: in the last 60 completed runs the **Filings node made an LLM
call in only 8** (the other 52 returned "no chunks retrieved" in ~4 s and the run was
marked degraded). The live view should make this kind of systemic soft failure
unmissable (Health tab: "filings skipped in 52 of 60 runs").

## §3 Target design

### 3.1 Data model (schema v9, forward-only migration)

```sql
-- node_runs: two-phase. INSERT at node entry (status='running', ended_at NULL),
-- UPDATE at exit. Readers that count statuses must tolerate 'running'.
-- new column:  node_runs.pid INTEGER            -- which process ran it (stale detection)

-- graph_runs: new column  source TEXT DEFAULT ''  -- 'dashboard' | 'cio' | 'telegram' | 'backtest' | 'cli'

-- llm_calls: two-phase. INSERT at request start with the full prompt, UPDATE on response.
-- new columns: status TEXT DEFAULT 'completed'   -- 'running' | 'completed' | 'failed'
--              ended_at TEXT, error TEXT, attempt INTEGER DEFAULT 1

CREATE TABLE prompt_blobs (            -- content-addressed, deduplicated text
  sha256      TEXT PRIMARY KEY,
  content     TEXT NOT NULL,
  n_chars     INTEGER NOT NULL,
  first_seen  TEXT NOT NULL,
  last_seen   TEXT NOT NULL,
  seen_count  INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE llm_messages (            -- the full messages of one call, in order
  call_id   INTEGER NOT NULL REFERENCES llm_calls(id),
  seq       INTEGER NOT NULL,          -- 0 = system, 1 = user, 2 = assistant (response)
  role      TEXT NOT NULL,
  sha256    TEXT NOT NULL REFERENCES prompt_blobs(sha256),
  PRIMARY KEY (call_id, seq)
);
CREATE TABLE spans (                   -- steps of non-graph flows (CIO now; Discovery / ad-hoc later)
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_kind TEXT NOT NULL,           -- 'cio_run' (later: 'discovery_run', ...)
  parent_id   TEXT NOT NULL,           -- cio_runs.run_id
  name        TEXT NOT NULL,           -- 'probe' | 'gates' | 'decide' | 'budget' | 'drill' | 'summary' | 'notify'
  label       TEXT,                    -- e.g. the ticker
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  status      TEXT NOT NULL,           -- running | completed | failed
  meta_json   TEXT,                    -- small: counts, decision, drill_run_id
  error       TEXT
);
```

Rules:

- `_connect()` sets `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`.
- `prune_blobs(days=PROMPT_RETENTION_DAYS)` deletes `llm_messages` rows older than the
  window and any blob no row references. Excerpts in `llm_calls` stay. Called from
  `init_db` at most once a day (meta key `last_prune`).
- `sweep_stale_runs(minutes=30)` marks `graph_runs` / `node_runs` / `llm_calls` /
  `spans` rows still `running` whose process is gone (`pid` not alive) or older than the
  window as `failed` with `error='stale: process ended without finishing'`. Called by the
  Live tab on each poll (cheap) and by the CIO dispatcher at cycle start.
- System prompts dedupe naturally (same sha256 every run); user prompts and responses are
  unique per call. Expected growth ≈ 0.3–1 MB per drill-in; the window bounds it.

### 3.2 Instrumentation points

| File | Change |
|---|---|
| `agents/__init__.py::_safe_node` | `start_node_run()` on entry → `finish_node_run()` in `finally`. `invoke_with_telemetry(graph, state, *, source)`. |
| `utils/openrouter.py` interceptor | Two-phase `llm_calls` write; store full messages via `record_llm_messages(call_id, messages, response)`; keep excerpts for the old readers. |
| `cio/cio.py` | `with state_db.span('cio_run', cio_run_id, 'probe', label=ticker): …` around each step; planner accumulator named `cio.planner` with `current_run_id = cio_run_id`. Drills already link through `cio_actions.drill_run_id`. |
| `ui/_runner.py`, `data/telegram.py`, `backtest/runner.py`, `scripts/*` | Pass `source=`. |
| `data/state.py` | Schema v9, pragmas, `live_snapshot()`, `run_snapshot(run_id)`, `prune_blobs()`, `sweep_stale_runs()`, `node_medians(limit=60)`. |

### 3.3 The snapshot (one shape for live and replay)

```python
{
  "now": "...",                       # server time, so the client can offset its clock
  "poll_s": 1,
  "vitals": {"runs_today": 3, "cost_today_usd": 0.10, "failed_24h": 0, "degraded_24h": 3,
             "active": 1, "next_heartbeat": "2026-10-17T05:00-07:00"},
  "medians": {"fundamentals": 97.7, "filings": 10.1, "news": 21.0, "risk": 96.5,
              "synthesis": 12.0, "run": 214.2},
  "cio": [ {"run_id": "...", "trigger": "heartbeat", "started_at": "...", "status": "running",
            "steps": [ {"name": "decide", "label": "GVA", "started_at": "...", "ended_at": null,
                        "status": "running", "meta": {...}} ],
            "actions": [ {...cio_actions row...} ] } ],
  "runs": [ {"run_id": "...", "ticker": "GVA", "thesis": "construction", "source": "cio",
             "status": "running", "started_at": "...", "ended_at": null,
             "nodes": [ {"node": "fundamentals", "status": "running", "started_at": "...",
                         "ended_at": null, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                         "calls": [ {"id": 1320, "model": "...", "status": "running",
                                     "started_at": "...", "tokens_in": null, ...,
                                     "messages": [{"role": "system", "text": "..."},
                                                  {"role": "user", "text": "..."}],
                                     "response": null} ] } ],
             "errors": [] } ],
  "recent": [ {...last 20 graph_runs with per-node status...} ]
}
```

Live includes full messages only for **active** runs; the Runs tab loads a past run's
messages on demand. The Live fragment hashes the snapshot (without `now`) and only calls
`components.html` with new HTML when the hash changes; otherwise it re-emits the previous
HTML string, which Streamlit does not re-mount. Elapsed counters and progress rings tick
client-side from timestamps.

### 3.4 Page structure

```
ui/pages/mission_control.py        tabs: Live | Runs | Health
  Live    st.fragment(run_every="1s") → data.state.live_snapshot() → ui/mc_live.html template
  Runs    run picker (newest first, ?run_id= honoured) → header, waterfall, agent cards,
          LLM call trace with FULL system / user / output (st.expander + st.code),
          errors, payloads, CIO lineage  (today's Run Inspector, moved)  + "Replay" →
          the same component in replay mode with a scrubber
  Health  today's panels, unchanged: freshness, filings manifest, runs table, daily cost,
          CIO actions / cycles / model performance, errors, Notion, eval runs, cached demos
ui/pages/run_inspector.py          → st.switch_page("pages/mission_control.py") keeping run_id
ui/mc_live.html                    the component: one file, vanilla JS, no CDN, ~800 lines
ui/components.py                   shared palette already there; add `mc_tokens()` for the template
```

## §4 Mockups

Three layouts over the same snapshot. All three show: the execution graph, which node is
running, system and user prompt and output per node, telemetry per node and per run, the
CIO cycle that triggered the run, and replay of a past run.

| | A · Assembly line | B · Flight recorder | C · Control room |
|---|---|---|---|
| Metaphor | adlc's line: stations in topology order, runs as crates moving along a belt | Observability tool: waterfall of spans with a playhead, event log, span drawer | Animated graph canvas: icon nodes, labelled edges that light up as data flows, CIO step ribbon above, fleet column beside |
| Best at | several runs at once; "where is each run" | timing, cost, slow-node diagnosis, exact replay | the mechanism and the current moment; explaining the system to a visitor |
| Weak at | timing precision; the fan-out is a column, not a graph | spatial sense of the graph | many concurrent runs (one canvas = one run; others queue in the fleet column) |
| Prompts in / out | bottom tray with tabs | right drawer with tabs | right drawer with tabs |
| Component effort | medium (~700 lines) | medium (~800 lines) | higher (~900 lines, SVG layout) |

**Decision (2026-10-16):** C for the Live tab, B's waterfall and drawer for the Runs tab.
They share the snapshot and the inspector drawer, so the second view costs roughly a third
of the first. A is not built.

## §5 Phases and acceptance tests

Each phase is a PR, gated on its test plan passing end to end and your explicit
"proceed" (CLAUDE.md §16.5).

### Phase 1 — telemetry foundation (schema v9), ~5 h

Deliverables: `data/state.py` v9 + pragmas + sweeper + pruning + snapshot builders;
two-phase `_safe_node`; two-phase interceptor with full messages; CIO spans; `source`.

Tests (`pytest`):

- `test_state.py`: v8 → v9 migration on a copy of a populated DB keeps every row; new
  tables exist; pragmas applied; `start_node_run` → `finish_node_run` round trip; a
  running node is visible to `live_snapshot()` between the two; `prune_blobs` removes
  only unreferenced blobs older than the window; `sweep_stale_runs` closes a `running`
  row whose pid is dead and leaves a live one alone.
- `test_openrouter_telemetry.py`: the call row exists with `status='running'` and the
  full system + user text **before** the SDK returns (use a blocking stub); after the
  response it has tokens, cost, latency and the assistant message; the system blob is
  stored once across two calls; a failed call ends `failed` with the error and re-raises;
  excerpts still populated for old readers.
- `test_graph_telemetry.py`: one `node_runs` row per node, `running` while inside the
  node, `completed` after; `source` persisted; parallel workers each see their own row.
- `test_cio_workflow.py`: a cycle writes spans `probe, gates, decide, budget, drill,
  summary, notify` in order, the `drill` span's `meta.drill_run_id` matches
  `cio_actions.drill_run_id`, planner calls carry `node='cio.planner'` and
  `run_id=cio_run_id`.
- Manual: run a drill from the dashboard and `SELECT node, status FROM node_runs WHERE
  run_id=…` mid-run shows the running node; the `.db` size delta for one run is logged.

### Phase 2 — Mission Control page with Live + Runs tabs, ~10 h

Deliverables: the chosen mockup as `ui/mc_live.html`; tabs; Run Inspector moved into
Runs with full prompts; replay; redirect stub; smoke tests.

Tests:

- `test_dashboard_smoke.py`: page renders with an empty DB and with a populated one; the
  Live tab emits exactly one `components.html` per fragment run; identical snapshots
  produce byte-identical HTML (no re-mount); a changed snapshot produces new HTML; the
  Runs tab shows the full system and user prompt for a call; `?run_id=` preselects the
  run on both the new page and the redirect stub; Health tab panels unchanged (existing
  tests keep passing).
- Component (node, no browser): a small `tests/mc_live/` harness that loads the template
  with a fixture snapshot in `jsdom` is **out of scope** (no Node toolchain in the repo);
  instead a pytest check that the template parses as HTML and every `{{token}}` is
  substituted.
- Manual: a live drill from the dashboard, a live CIO on-demand cycle
  (`python -m cio.dispatcher --mode on_demand --ticker CAT`), and a replay of an old
  run, each screenshotted and attached to the PR.

### Phase 3 — polish and docs, ~4 h

ETA from medians, Health tab "systemic soft failures" card, keyboard navigation in the
component, `docs/ARCHITECTURE.md` §11 revision (v9, two-phase writes, snapshot
contract, iframe re-mount rule) and §10.3 (page table), `docs/POSTPONED.md` cleanup
(Run Inspector rows), CLAUDE.md §16.7 update, README screenshot.

## §6 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Iframe re-mount flashes on each state change | Re-mount only on snapshot change (≈15 per run); the component restores scroll, selection and tab from `sessionStorage` keyed by run id. |
| `database is locked` with three processes | WAL + `busy_timeout`; every write is one short transaction; the reader never holds a transaction. |
| Prompt storage growth | Dedupe + retention; log the size delta per run in Phase 1; `PROMPT_RETENTION_DAYS` in `.env`. |
| Streamlit `AppTest` and fragments | `run_every` is inert under `AppTest`; tests drive the fragment function directly. |
| Branch collision with `feat/discovery-graph-viz` (PR #8, open and mergeable; touches `docs/ARCHITECTURE.md`, `docs/POSTPONED.md`, `tests/test_dashboard_smoke.py`) | Decided: merge PR #8 before Phase 2; Phase 1 keeps its ARCHITECTURE edits local to §11. |
| Deep links from Telegram | `data/telegram.py` builds dashboard URLs from `STREAMLIT_PUBLIC_URL`; the redirect stub keeps old links working. |

## §7 Out of scope (tracked in POSTPONED when approved)

Discovery / ad-hoc / Telegram router / Q&A / backtest spans (same `spans` table, later);
acting from the page (re-run a node, cancel a run); LangSmith read-back; a separate
server or port; per-attempt visibility of SDK retries (needs an httpx hook).

## §8 Decisions (taken 2026-10-16)

1. Live tab layout: **C** (control room graph); Runs tab: **B**'s waterfall and drawer.
2. Theme: **Day** (CLAUDE.md §13 palette).
3. Poll interval: **1 s**.
4. Prompt retention: **90 days** (`PROMPT_RETENTION_DAYS`).
5. Branch order: **merge PR #8** (`feat/discovery-graph-viz`) before Phase 2.
