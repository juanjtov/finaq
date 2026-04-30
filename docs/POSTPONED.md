# POSTPONED — items deferred but tracked

Living doc. Updated whenever we make a "not now, later" call. When something
ships, move it out of this file (don't strike-through — keep the doc lean).

Three categories:

- **§1 Scheduled** — has a step number in the build plan; will be built in order.
- **§2 If-needed** — deferred decisions with explicit triggers; revisit when the
  trigger fires, drop if it never does.
- **§3 Out of scope** — explicitly ruled out for the foreseeable future.

---

## §1 Scheduled (in the plan, not yet built)

### Phase 0 — drill-in completion

| Step | What | Status | Notes |
|---|---|---|---|
| 5z | **Observability foundation** | pending | LangSmith env opt-in, `data/state.py` SQLite layer, `_safe_node` wires `record_*` calls. Mission Control reads its placeholders today; Step 5z will populate them. |

### Phase 1 — personal MVP

| Step | What | Notes |
|---|---|---|
| 9  | **Notion memory layer** | `data/notion.py`, 4 DBs (theses/reports/alerts/watchlist), one-way sync (read notes / write reports + alerts). |
| 10 | **Telegram bot (bidirectional)** | Slash commands + Haiku NL routing. `/drill`, `/analyze` (ad-hoc thesis Discovery-lite), `/scan`, `/note`, `/thesis`, `/status`, `/help`. Allowlist enforced. |
| 11 | **Triage agent + scheduling** | Real Haiku-backed continuous Triage. `launchd` first, droplet `systemd timer` after Step 12. Healthchecks.io ping after each run. |
| 12 | **Droplet deployment** | DigitalOcean $6/mo droplet, Caddy HTTPS, three systemd units, `deploy/install.sh`. Supersedes the Cloudflare Tunnel option. |

### Mission-control surfaces (each unlocked by an existing build step)

| Surface | Lands in step | What it shows |
|---|---|---|
| LangSmith tracing | 5z | Per-LLM-call latency, tokens, full prompt/response, replay. |
| Streamlit Mission Control page | 8 | Graph-run history, daily-cost chart, freshness cards, recent alerts, errors. |
| Telegram `/status` command | 10 | Triage last-run, alerts in last 24h, today's $ spend, open errors. |
| Healthchecks.io ping | 11 | External liveness watcher. Fires if Triage stops emitting. |
| `journalctl` on the droplet | 12 | SSH-time deep dive into systemd unit logs. |

---

## §2 If-needed — deferred decisions with explicit triggers

Each item has a **trigger**: the observable condition that should cause us to
revisit. If the trigger never fires, we never build it.

### RAG / retrieval enhancements (within Step 5b's scope)

| Item | Trigger | Estimated effort |
|---|---|---|
| **Cross-encoder re-ranking** (`sentence-transformers/ms-marco-MiniLM`) | Filings citations are noticeably low-quality after Step 8 demo, OR Synthesis bull/bear bullets cite stale chunks despite freshness markers. | 1h + ~500MB local model + 2–4s/query |
| **MMR (Maximum Marginal Relevance)** | Top-K returned chunks are clearly redundant (e.g., 5 of 8 chunks come from the same paragraph). | ~30 lines, no new dep |
| **Sentence-window retrieval** | LLM Filings synthesis quotes truncated mid-sentence or misses surrounding context. | ~20 lines |
| **LLM-generated subqueries** (cheap-tier model rewrites the 3 subquery templates per thesis × ticker) | Ad-hoc `/analyze` (Step 10) or new theses produce shallow Filings synthesis with hardcoded templates. | 1 extra cheap-tier call per drill-in, ~$0.001 |
| **Hybrid corpus expansion** (BM25 over the *full* ticker corpus, not just the 60 semantic candidates) | A known-relevant chunk is missed because it's outside the candidate pool. | Re-architect `query()` to fetch the whole filtered corpus first |
| **Chunk-size tuning** (currently 800 tokens) | Sonnet's synthesis cites mid-sentence-cut chunks OR top-8 are clearly redundant from the same paragraph. | One-line constant change + re-ingest |

### RAG evaluation enhancements (within Tier 1/2/3 already shipped — covers Filings AND News)

| Item | Trigger | Estimated effort |
|---|---|---|
| **`pytest -m eval` nightly on droplet** (Tier 2 + Tier 3 run automatically each night for *every* agent with an eval suite — Filings, News, future Risk/Synthesis. Results persisted to `state.db`, Mission Control shows trend) | After Step 12 droplet deploy, when historical eval data becomes valuable (a few days after deploy). | systemd timer + state.db schema |
| **Reference-answer generation for context_recall** (Tier 3 currently runs without ground_truth; adding it unlocks `context_recall` metric) | When we want to compare retrieval strategies head-to-head and `context_precision` alone isn't discriminative enough. | Curate reference answers per golden query (~1h) |
| **Cost-tracking for eval runs** | When monthly eval cost crosses ~$10/mo and we want to budget. | Persist per-run token + dollar cost to `state.db.eval_runs` |
| **Multi-ticker golden set** (Filings: NVDA-only; News: NVDA-only) | When ANET / AVGO / other tickers are ingested at scale and we need per-ticker quality bars. | Add 5–8 queries per ticker in `tests/eval/{filings,news}_golden_queries.py` |
| **Synthesis-agent eval suite** (Tier 1+2+3 mirror, once Step 7 lands) | When Step 7 ships the LLM synthesis. | Mirror existing suites. |

### Valuation enhancements (within Step 6 already shipped)

| Item | Trigger | Estimated effort |
|---|---|---|
| **Live sector-P/E data feed** (replaces hardcoded `data/sector_multiples.json`) | When a free reliable source is identified (e.g., a paid Damodaran API, an FRED-style aggregator, or scraping Yahoo Finance sector pages programmatically) | Provider-research + new module + cache; ~3h |
| **Working capital changes in owner earnings** | When backtesting reveals consistent over- or under-valuation traceable to WC dynamics | Add `Δ-WC as % of revenue` parameter to `Projections`; small math change in MC |
| **Correlation between MC parameters** (growth ↔ margin ↔ multiple) | When backtesting reveals our independence assumption is materially wrong, OR when Synthesis reports feel under-uncertain | Replace `np.random.normal` per-param with multivariate-normal using a covariance matrix from historical data; ~30 lines |
| **Per-period growth path** (year 1 of 50% taper to year 5 of 10%) | When user wants to model deceleration explicitly rather than via single CAGR | Refactor MC to take `growth_path: list[float]` rather than scalar; ~50 lines |
| **Sensitivity analysis output** (∂P50/∂growth, ∂P50/∂margin, etc.) | When user wants to know "which input matters most" for a given drill-in | Compute partial derivatives via finite differences; render in Synthesis report |
| **Backtesting** (replay historical projections through MC, check actual vs predicted) | When we have ≥1 year of drill-in history saved | Big — needs historical Fundamentals projections snapshotted; ~1 day |
| **Risk widens Monte Carlo stds (Approach B)** | Synthesis reports feel under-uncertain after Step 8 demo, OR a missed catastrophic scenario causes a real loss. | High `level` would multiply MC stds by ~1.5x; ~30 lines of code, requires updating Synthesis to surface the widened distribution. |
| **Risk injects tail-risk in Monte Carlo (Approach C)** | When Approach B isn't enough — the real risk profile has fat tails (e.g., regulatory shutdown). | Mixture distribution in MC: 95% normal model + 5% catastrophic mode. Bigger refactor; ~1h. |

### Fundamentals agent (within Step 5a's scope, may revisit when Step 5d lands)

| Item | Trigger | Estimated effort |
|---|---|---|
| **Per-thesis Buffett threshold cutoffs** (e.g., AI cake gets `fcf_yield > 4`, Construction gets `> 7`) | Universal cutoffs fire too aggressively for one thesis (e.g., every AI cake ticker triggers `fcf_yield < 4` overvaluation flag, drowning the signal). | Schema field on Thesis; ~10 lines |
| **Moat metadata on Thesis schema** (Buffett option A from earlier discussion) | Synthesis reports feel too quantitative — missing durable-advantage framing. | Schema change + per-ticker `moat_grade`/`moat_note` annotation; ~30 min |
| **Capital-allocation thresholds** (Buffett option C: Form 4 insider buys, buybacks-at-value, dividend hikes) | Management actions become decision-relevant signal we keep manually noticing. | Requires Form 4 ingestion (new EDGAR pipeline); ~3h |
| **Robust capex/revenue alignment for non-Dec fiscal years** | A non-calendar-FY ticker (e.g., NVDA's Jan FYE) shows weirdly low/high `capex_to_revenue_5yr_avg`. | Currently matches by `YYYY` prefix only — switch to fiscal-period-aware joining; ~30 min |
| **LLM cost-source consolidation** (drop the static `MODEL_PRICING` table once LangSmith is the source of truth) | LangSmith dashboard proves accurate enough that maintaining the local price table feels redundant. | Remove a few dozen lines; trigger lives in Step 5z |

### Theses content

| Item | Trigger | Notes |
|---|---|---|
| **Halo · NVDA universe expansion to include NVDA explicitly** (8 tickers instead of 7) | The relationship validator wants `from: "NVDA"` edges, OR Triage misses an NVDA-only signal because NVDA isn't in Halo's universe. | One-line JSON edit. |
| **Construction thesis: post-IRA threshold language** | The Inflation Reduction Act ages out of materiality (next-administration changes). | Replace `filing_mentions contains "Inflation Reduction Act"` with whatever's currently driving the buildout. |
| **Halo · NVDA: generation-agnostic phrasing** (drop "Blackwell"-specific threshold) | NVIDIA ships next-generation silicon and our threshold goes stale. | Change `filing_mentions contains "Blackwell"` to a more durable signal name. |

### Phase 2+ (post-Phase 1)

| Item | Trigger / phase | Notes |
|---|---|---|
| **Full Discovery agent + halo graph builder** | Phase 2 | Step 10's `/analyze` is the Discovery-lite stand-in; persistent halo graph is the Phase 2 piece. |
| **Bidirectional Notion sync** (your edits → re-fetched as inputs) | Phase 2 | Phase 1 is one-way (read notes, write reports/alerts) only. |
| **Synthesis cycle-based re-trigger** (planner-style "ask for more" loop) | Phase 2+ | When `state.gaps` contains the same item across many runs, OR a single drill-in materially needs an upstream re-query (e.g. user said "expand on point 3 in the report"). Tradeoffs are documented in ARCHITECTURE §6.18 — breaks DAG topology, blows the <5min target, risks unbounded cost. We built the `gaps: list[str]` observability path instead so we have data on whether the loop would have fired. Phase 2 trigger: ≥30% of runs surface a `gaps` entry that maps cleanly to a known upstream re-query (e.g. "no segment-level capex split" → re-run filings with different subquery). |
| **Pattern detection** (cross-thesis multiplicity scoring) | Phase 3 | The `VRT/CEG/ANET/PWR` overlap is currently data only — automated weighting comes in Phase 3. |
| **Threshold learning** | Phase 3 | Material thresholds are hand-set in JSON until then. The mechanism would observe which alerts you drilled in vs ignored. |
| **Backtesting** | Phase 3 | Replay historical filings/news through the system to validate alert quality. |

---

## §3 Out of scope (explicitly not building)

These have been considered and ruled out. Re-opening any requires a deliberate
re-decision, not a drift.

| Item | Why ruled out |
|---|---|
| **Grafana / Prometheus / Sentry / DataDog** | Single-user single-box system; LangSmith + SQLite is sufficient. Adding industrial-grade observability here is not justified by the operational surface. |
| **LangGraph Cloud / Platform deployment** | Local-first per CLAUDE.md §2. `langgraph dev` runs locally; cloud deployment is the only outbound option ruled out. |
| **Cloudflare Tunnel for Streamlit reachability** | Dropped from Step 12 once the user committed to the droplet. Tunnel is a "laptop-awake" stopgap; droplet is the real answer. |
| **Multi-tenant / multi-user** | FINAQ runs as `juan`. No auth, no users table. Ever. |
| **Hosted vector DB** (Pinecone, Weaviate Cloud) | ChromaDB on local disk works at our scale. Cloud vector DB adds bill, latency, dependency. |
| **Cron / Celery / Airflow** | Two scheduled jobs (Triage, daily backups) don't justify a workflow engine. systemd timers are the right tool. |
| **Pyproject.toml as dependency manifest** | Replaced by `requirements.txt` per CLAUDE.md §16.3. `pyproject.toml` is kept for tool config only. |

---

## How to update this doc

When something ships:
1. Delete the row from §1 or §2 (don't strike-through — keep the file scannable).
2. If new "if-needed" items emerged during the work, add them to §2 with an explicit trigger.

When a §2 trigger fires:
1. Move the item to §1 with a step number and target.
2. Add it to the corresponding step's task description in the build plan.

When making a "not now" call during a session:
1. Land the item in §2 with a one-line trigger.
2. Mention it in the chat summary so it's not silently lost.
