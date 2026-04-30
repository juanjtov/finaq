# FINAQ — personal equity research advisor

Single-user, multi-agent system that monitors SEC filings + news against
your hand-written investment theses, and on demand produces an
institutional-grade drill-in report (Fundamentals + Filings + News → Risk
→ Monte Carlo → Synthesis) plus a fair-value distribution.

Built on LangGraph, OpenRouter, ChromaDB, Streamlit. See
[`CLAUDE.md`](./CLAUDE.md) for the build spec and
[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for decision history.

## Quick start

```bash
# 1. Install deps (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure keys + models
cp .env.example .env
$EDITOR .env
#   → fill in OPENROUTER_API_KEY, TAVILY_API_KEY, SEC_EDGAR_USER_AGENT
#   → MODEL_* defaults are 2026-04-26 OpenRouter latest; swap any to swap models

# 3. Ingest the AI cake universe (one-time, ~10 min)
python scripts/ingest_universe.py

# 4. Launch the dashboard
streamlit run ui/app.py
```

Open <http://localhost:8501>. Pick a thesis, enter a ticker, hit
**Run drill-in**. First run takes 3–5 min on a real ticker; subsequent runs
on the same ticker × thesis are cached in `data_cache/demos/` and load
instantly.

## What's in the dashboard

| Page | What it does |
|---|---|
| **Dashboard** (`ui/app.py`) | Main drill-in view: synthesis report, MC histogram, per-agent expanders, PDF download. |
| **Direct Agent** | Talk to one agent at a time. Re-run a single step or ask a free-text question scoped to that agent. |
| **New thesis** | Form-based thesis creator. Validates via Pydantic, writes to `theses/<slug>.json`. |
| **Architecture** | Live snapshot of the platform: LangGraph topology, agent cards (live model strings), data sources, test tiers, palette. |
| **Methodology** | Per-thesis valuation parameters with rationale, Owner-Earnings DCF formula, full `FINANCE_ASSUMPTIONS.md` rendered. |
| **Mission Control** | Eval run history, data-source freshness, cached drill-in audit. |

## Running the test suite

```bash
# Default — fast (no external calls, no LLM cost)
pytest

# Real-API integration tests (yfinance, EDGAR, Tavily, OpenRouter)
pytest -m integration

# LLM-judge eval (~$0.025/run)
pytest -m eval
```

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) §7 for the 3-tier eval
pattern. Tests are gated; never `xfail`-ed to advance per CLAUDE.md §16.5.

## Repository layout

```
agents/        one file per agent — fundamentals, filings, news, risk, synthesis, qa
  prompts/     system prompts for each agent (.md)
data/          edgar.py, yfin.py, chroma.py, tavily.py, treasury.py
theses/        hand-written thesis JSONs (Phase 0)
ui/            Streamlit app + pages
utils/         schemas, monte_carlo, charts, pdf_export, models, openrouter
tests/         Tier 1 (deterministic) + Tier 2 (LLM-judge) + Tier 3 (integration)
docs/          ARCHITECTURE.md (decisions), FINANCE_ASSUMPTIONS.md (math), POSTPONED.md (deferred)
scripts/       ingest_universe.py, run_telegram_bot.py (Phase 1), run_triage.py (Phase 1)
data_cache/    gitignored runtime cache (edgar/, yfin/, chroma/, demos/, eval/, fixtures/)
.streamlit/    Streamlit theme config
.env.example   key + model env-var template
```

## Phase scope

**Phase 0 — Drill-in (current).** Streamlit dashboard, hybrid Owner-Earnings
DCF + Multiple Monte Carlo, 9-section synthesis report + PDF, three
hand-written theses (`ai_cake`, `nvda_halo`, `construction`), per-agent
direct invocation + Q&A, form-based thesis creator.

**Phase 1 — Personal MVP.** Notion memory, bidirectional Telegram bot
(per-agent slash commands mirror the dashboard's Direct Agent), continuous
Triage agent, DigitalOcean droplet deployment.

**Phase 2+ (deferred).** Full Discovery agent with halo graph, bidirectional
Notion sync, pattern detection / cross-thesis multiplicity, Synthesis
cycle-based re-trigger. See [`docs/POSTPONED.md`](./docs/POSTPONED.md) for
the full list with explicit triggers.

## License

See [`LICENSE`](./LICENSE).
