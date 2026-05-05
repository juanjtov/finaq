# CLAUDE.md — FINAQ Build Spec

> Read this file first. Every prompt to you should assume this file is loaded.
> See `FINAQ_Context.docx` for the high-level pitch and target user.

## 1. Project context

FINAQ is a **personal** equity research advisor for a single user (no auth, no multi-tenancy). It runs continuously on a small server, monitors SEC filings and news against the user's three theses, pushes only material signals to Telegram, and on demand produces an institutional-grade investment report in under five minutes via a multi-agent drill-in flow plus a Monte Carlo fair-value distribution.

The hackathon (Phase 0) builds the **drill-in + Monte Carlo + Streamlit UI** at full quality with 2–3 pre-built thesis JSONs as a stand-in for the Discovery agent, and a manual "Run scan" button as a stand-in for continuous Triage. Everything else is post-hackathon work — see `§14 Out of scope`.

## 2. Architecture decisions

These are non-negotiable for the hackathon build. Do not invent alternatives without asking.

- **Single-user, no auth.** The whole system runs as `juan`. No login, no users table.
- **All LLM calls go through OpenRouter** using its OpenAI-compatible endpoint (`base_url="https://openrouter.ai/api/v1"`). Never call `anthropic` or `openai` SDKs directly.
- **LangGraph for orchestration.** Use `StateGraph` with a TypedDict state. Parallel branches via separate edges from a single source node, joined by a `join` reducer. No CrewAI, no AutoGen.
- **ChromaDB on local disk** for filings RAG. Persistent collections, one per ticker. Embeddings via OpenRouter's `text-embedding-3-small` (or whichever embedding model is available — confirm at build time).
- **Notion as memory of record (post-hackathon).** For Phase 0, write to local `JSON` files in `./data_cache/memory/` with the same shape Notion will eventually hold.
- **Telegram for alerts (post-hackathon).** For Phase 0, alerts surface inside the Streamlit dashboard only.
- **Streamlit for the demo UI.** Don't reach for FastAPI + React. The demo runs locally.
- **Local-first.** The hackathon build runs on the developer's laptop. Do not introduce hosted dependencies (Cloud Run, Vercel, etc.). `langgraph dev` for LangGraph Studio is local-only and explicitly allowed; LangSmith tracing (opt-in via `LANGSMITH_TRACING=true`) is the only outbound observability dependency permitted in Phase 0. Phase 1 hosting on a single DigitalOcean droplet is the eventual deployment target — single-box, single-tenant, no managed services beyond what's in `requirements.txt`.

## 3. Model routing

All model strings below are passed as the `model` argument to OpenRouter. Confirm exact strings at build time by hitting `https://openrouter.ai/api/v1/models` — these reflect current naming convention.

| Agent | Default model (as of 2026-04-26) | Env var | Rationale |
|---|---|---|---|
| Triage | `anthropic/claude-haiku-4.5` | `MODEL_TRIAGE` | Cheap, fast, called frequently to filter noise |
| Fundamentals | `anthropic/claude-sonnet-4.6` | `MODEL_FUNDAMENTALS` | Numeric reasoning, modest context |
| Filings | `anthropic/claude-sonnet-4.6` | `MODEL_FILINGS` | RAG synthesis |
| News | `anthropic/claude-sonnet-4.6` | `MODEL_NEWS` | Catalyst extraction |
| Risk | `anthropic/claude-sonnet-4.6` | `MODEL_RISK` | Reads other workers' outputs only |
| Synthesis | `anthropic/claude-opus-4.7` | `MODEL_SYNTHESIS` | Final thesis writing, highest quality |
| Router (Phase 1) | `anthropic/claude-haiku-4.5` | `MODEL_ROUTER` | Telegram intent classification, must be cheap + fast |
| Ad-hoc Thesis (Phase 1) | `anthropic/claude-sonnet-4.6` | `MODEL_ADHOC_THESIS` | Decomposes a free-text topic into a thesis JSON for `/analyze` |
| Judge (RAG eval) | `anthropic/claude-haiku-4.5` | `MODEL_JUDGE` | Used by `pytest -m eval` to score retrieval relevance and faithfulness; cheap-tier role |
| Per-agent Q&A | `anthropic/claude-haiku-4.5` | `MODEL_AGENT_QA` | Powers `agents/qa.py.ask()` for the dashboard's Direct Agent panel + Phase 1 Telegram per-agent commands. Cheap-tier role since calls are frequent and scoped to a single agent's structured output. |
| Discovery (Phase 2+) | `anthropic/claude-opus-4.7` | `MODEL_DISCOVERY` | Thesis decomposition, runs rarely |
| Embeddings | `text-embedding-3-small` (via OpenRouter) | `MODEL_EMBEDDINGS` | ChromaDB filings collection |

Hard rule: **never hardcode model strings inside agent files**. Model strings live in `.env` (placeholders in `.env.example`); `utils/models.py` is a thin typed registry that reads them via `os.getenv()` and exposes typed constants. Agents import only from `utils/models.py`. Swapping a model is a one-line `.env` change, not a code change.

## 4. Repository structure

```
/finaq
  /agents                  # one file per agent, each exports an async run() coroutine
    fundamentals.py
    filings.py
    news.py
    risk.py
    synthesis.py
    triage.py              # Phase 1+; Phase 0 has a stub that returns {} 
    discovery.py           # Phase 2+; Phase 0 absent
  /data
    edgar.py               # SEC EDGAR fetcher
    yfin.py                # yfinance wrapper
    chroma.py              # ChromaDB ingest + query
    __init__.py
  /theses                  # hand-written JSONs for Phase 0
    ai_cake.json
    nvda_halo.json
    construction.json
  /utils
    models.py              # OpenRouter model strings
    openrouter.py          # OpenRouter client factory
    state.py               # LangGraph TypedDict state
    monte_carlo.py
    pdf_export.py
    schemas.py             # Pydantic models for thesis JSON, agent outputs
  /ui
    app.py                 # Streamlit entrypoint
    components.py          # reusable Streamlit widgets
  /data_cache              # gitignored; created at runtime
    /edgar/{ticker}/       # downloaded filings
    /chroma/               # ChromaDB persistent dir
    /memory/               # JSON files standing in for Notion
  CLAUDE.md
  README.md
  pyproject.toml
  .env.example
  .gitignore
```

## 5. Environment setup

Python 3.11+. **`requirements.txt` is the dependency manifest** — every new top-level import must be added there in the same change. `pyproject.toml` is kept *only* for tool config (ruff / black / pytest), not for deps. Install via `pip install -r requirements.txt` (or `uv pip install -r requirements.txt` if `uv` is available).

Required packages (Phase 0): `langgraph`, `langchain-openai`, `openai` (for the OpenAI-compatible OpenRouter client), `chromadb`, `yfinance`, `sec-edgar-downloader`, `streamlit`, `numpy`, `pandas`, `reportlab`, `pydantic`, `python-dotenv`, `tavily-python`, `tenacity`, `httpx`, `matplotlib`.

Phase 1 additions: `notion-client`, `python-telegram-bot`.

`.env.example` must contain:
```
# API keys
OPENROUTER_API_KEY=sk-or-v1-...
TAVILY_API_KEY=tvly-...
SEC_EDGAR_USER_AGENT="FINAQ/0.1 juan@example.com"

# Model strings — swap any of these to swap models. Defaults are 2026-04-26 latest.
MODEL_TRIAGE=anthropic/claude-haiku-4.5
MODEL_FUNDAMENTALS=anthropic/claude-sonnet-4.6
MODEL_FILINGS=anthropic/claude-sonnet-4.6
MODEL_NEWS=anthropic/claude-sonnet-4.6
MODEL_RISK=anthropic/claude-sonnet-4.6
MODEL_SYNTHESIS=anthropic/claude-opus-4.7
MODEL_ROUTER=anthropic/claude-haiku-4.5
MODEL_ADHOC_THESIS=anthropic/claude-sonnet-4.6
MODEL_EMBEDDINGS=text-embedding-3-small

# Phase 1 (uncomment when ready)
# NOTION_API_KEY=secret_...
# NOTION_DB_THESES=...
# NOTION_DB_REPORTS=...
# NOTION_DB_ALERTS=...
# NOTION_DB_WATCHLIST=...
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
# STREAMLIT_PUBLIC_URL=http://localhost:8501
```

`SEC_EDGAR_USER_AGENT` is required by the SEC; do not skip it.

## 6. Data layer

### 6.1 SEC EDGAR (`data/edgar.py`)

```python
async def download_filings(ticker: str, kinds: list[str] = ["10-K", "10-Q"], limit: int = 4) -> list[Path]:
    """Download recent filings for a ticker. Idempotent: skips if already on disk."""
```

Use `sec-edgar-downloader`. Save to `./data_cache/edgar/{ticker}/{accession}/full-submission.txt`. Always 2 most recent 10-Ks and 4 most recent 10-Qs per ticker.

### 6.2 yfinance (`data/yfin.py`)

```python
def get_financials(ticker: str) -> dict:
    """Returns {price_history_5y, income_stmt, balance_sheet, cash_flow, info}."""
```

Cache responses to `./data_cache/yfin/{ticker}.json` with a 24-hour TTL. yfinance is rate-limited and flaky — wrap in `tenacity` retries with exponential backoff.

### 6.3 ChromaDB (`data/chroma.py`)

**Chunking strategy:** split each filing on Item headers (`Item 1A. Risk Factors`, `Item 7. MD&A`, `Item 7A`, etc.). Within each section, split into ~800-token chunks with 100-token overlap. Each chunk's metadata: `{ticker, filing_type, accession, item, filed_date}`.

**Embeddings:** use OpenRouter's `/v1/embeddings` endpoint with `text-embedding-3-small` (model string in `MODEL_EMBEDDINGS`). Confirmed available 2026-04-26.

```python
def ingest_filing(ticker: str, filing_path: Path) -> int:
    """Chunk, embed via OpenRouter, write to collection 'filings'. Returns chunk count."""

def query(ticker: str, question: str, k: int = 8, item_filter: str | None = None) -> list[dict]:
    """Returns top-k chunks as [{text, metadata, score}, ...]."""
```

Single collection named `filings` with metadata-based filtering by ticker, not one collection per ticker — this makes cross-ticker queries trivial later.

## 7. Thesis JSON schema

Validated by Pydantic in `utils/schemas.py`. Same schema Discovery will emit in Phase 2.

```json
{
  "name": "AI cake",
  "summary": "Layered picks-and-shovels across silicon, hyperscaler, networking, power, and AI-native applications. Each layer has different durability and capex characteristics; the thesis is that the power layer is structurally undersupplied.",
  "anchor_tickers": ["NVDA", "MSFT"],
  "universe": ["NVDA", "AVGO", "TSM", "ASML", "MSFT", "GOOGL", "ORCL", "ANET", "VRT", "CEG", "PWR"],
  "relationships": [
    {"from": "NVDA", "to": "VRT", "type": "supplier", "note": "VRT supplies cooling for NVDA-spec racks"},
    {"from": "MSFT", "to": "CEG", "type": "customer", "note": "CEG signed PPAs with hyperscalers"}
  ],
  "material_thresholds": [
    {"signal": "data_center_capex_announcement", "operator": ">", "value": 5e9, "unit": "USD"},
    {"signal": "gross_margin_change_qoq", "operator": "abs >", "value": 200, "unit": "bps"},
    {"signal": "filing_mentions", "operator": "contains", "value": "capacity constraint", "unit": "text"}
  ]
}
```

Hand-write three of these for Phase 0: `ai_cake.json`, `nvda_halo.json`, `construction.json`. Tickers and relationships are listed in `FINAQ_Context.docx`.

## 8. LangGraph state machine

State (`utils/state.py`):

```python
from typing import TypedDict, Annotated
import operator

class FinaqState(TypedDict):
    ticker: str
    thesis: dict           # the loaded thesis JSON
    fundamentals: dict     # filled by Fundamentals agent
    filings: dict          # filled by Filings agent
    news: dict             # filled by News agent
    risk: dict             # filled by Risk agent
    monte_carlo: dict      # filled by MC engine
    report: str            # filled by Synthesis (markdown)
    messages: Annotated[list, operator.add]  # for streaming UI
```

Graph topology (`agents/__init__.py` exports `build_graph()`):

```
START → load_thesis → ┬→ fundamentals ─┐
                      ├→ filings        ├→ risk → monte_carlo → synthesis → END
                      └→ news ──────────┘
```

`fundamentals`, `filings`, `news` run in parallel (LangGraph fans out from `load_thesis`). `risk` joins them via a single edge that waits for all three. `monte_carlo` is a pure-Python node (no LLM) that consumes `fundamentals.projections`. `synthesis` is the only call to the synthesis-tier LLM (model resolved via `MODEL_SYNTHESIS`).

Use `graph.astream()` so the Streamlit UI can render node-by-node progress.

## 9. Agent specs

Every agent file exports `async def run(state: FinaqState) -> dict` returning a partial state update. Every agent reads `state["thesis"]` and conditions its prompt on the active thesis. System prompts live in `agents/prompts/` as plain `.md` files, loaded at import time.

### 9.1 Fundamentals (`agents/fundamentals.py`)

Inputs: `ticker`, `thesis`. Calls `data.yfin.get_financials()`, computes 5-year revenue CAGR, gross/operating margin trend, FCF trajectory, sector-relative valuation. LLM call summarizes findings and emits `projections = {revenue_growth_mean, revenue_growth_std, margin_mean, margin_std, exit_multiple_mean, exit_multiple_std}` for Monte Carlo. Output:

```python
{"fundamentals": {"summary": str, "kpis": dict, "projections": dict, "evidence": list[dict]}}
```

### 9.2 Filings (`agents/filings.py`)

RAG over the ChromaDB `filings` collection. Three subqueries per drill-in: risk factors, MD&A trajectory, segment performance. Each subquery returns top-8 chunks; LLM synthesizes a thesis-aware summary (e.g., "for the AI cake thesis, what does the latest 10-Q say about data-center demand?"). Output:

```python
{"filings": {"summary": str, "risk_themes": list[str], "mdna_quotes": list[dict], "evidence": list[dict]}}
```

`evidence` items must include `{accession, item, chunk_excerpt}` so the UI can render citations.

### 9.3 News (`agents/news.py`)

Tavily search for `{ticker} {company_name}` over the past 90 days. Top 15 results summarized into 5-7 catalysts and concerns, each tagged `bull|bear|neutral` and linked back to the source URL. Output:

```python
{"news": {"summary": str, "catalysts": list[dict], "concerns": list[dict], "evidence": list[dict]}}
```

### 9.4 Risk (`agents/risk.py`)

Reads `state.fundamentals`, `state.filings`, `state.news`. **No external calls** — purely a synthesis of red flags surfaced by the other three. The agent looks for four risk patterns:

1. **Convergent signals** — same risk surfaces in 2+ source agents (strongest signal)
2. **Threshold breaches** — `material_thresholds` from the thesis JSON that fired
3. **Divergent signals** — sources contradict each other (tension itself is a risk)
4. **Implicit gaps** — what the thesis assumes but the evidence doesn't support

The LLM emits a categorical `level` (LOW / MODERATE / ELEVATED / HIGH / CRITICAL); the composite `score_0_to_10` is *derived* from `level` via the canonical mapping `{LOW:2, MODERATE:4, ELEVATED:6, HIGH:8, CRITICAL:10}`. The categorical primary keeps the judgment model-stable (same anti-pattern fix as the LLM-judge prompt — see ARCHITECTURE §7.3); the integer score is kept for spec compatibility and quick visualisation. Risk does **not** modify Monte Carlo inputs in Phase 0 (see ARCHITECTURE §6.10).

Output:

```python
{
  "risk": {
    "level": str,                       # "LOW" | "MODERATE" | "ELEVATED" | "HIGH" | "CRITICAL"
    "score_0_to_10": int,               # derived from level
    "summary": str,                     # 3-5 sentences integrating the four risk patterns
    "top_risks": list[TopRisk],         # 3-7 items, each with title + severity 1-5 + explanation + sources
    "convergent_signals": list[dict],   # 0-5 items where ≥2 agents agreed
    "threshold_breaches": list[dict],   # 0-N items, one per fired threshold
    "errors": list[str],
  }
}
```

`TopRisk.sources` is the list of worker agents (`fundamentals` / `filings` / `news`) that surfaced the underlying signal — required for downstream traceability into Synthesis.

### 9.5 Synthesis (`agents/synthesis.py`)

Single LLM call (model resolved via `MODEL_SYNTHESIS`). Reads the entire state. Writes the final report as Markdown following the template in `§11`. Output:

```python
{"report": str}  # markdown string
```

## 10. Monte Carlo engine

Pure NumPy in `utils/monte_carlo.py`. Vectorized — no Python loops over draws.

```python
def simulate(
    revenue_now: float,
    growth_mean: float, growth_std: float,
    margin_mean: float, margin_std: float,
    exit_multiple_mean: float, exit_multiple_std: float,
    shares_outstanding: float,
    years: int = 5,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Returns:
        {
          "p10": float, "p25": float, "p50": float, "p75": float, "p90": float,
          "samples": np.ndarray  # shape (n_sims,) of fair value per share
        }
    """
```

Math: each sim draws `growth ~ N(growth_mean, growth_std)`, `margin ~ N(margin_mean, margin_std)`, `multiple ~ N(exit_multiple_mean, exit_multiple_std)`. Year-N revenue = `revenue_now * (1 + growth)^years`. Year-N earnings = `revenue_year_n * margin`. Fair value = `(earnings * multiple) / shares_outstanding`. Clip negative draws to zero (a growth rate of -150% is meaningless).

Render the histogram in the UI as a matplotlib `hist` with vertical lines at P10/P50/P90 and the current price marked.

## 11. Synthesis output format

The Synthesis agent must produce Markdown with **exactly these top-level sections**, in this order:

```markdown
# {TICKER} — {Thesis name} thesis update

**Date:** {YYYY-MM-DD} · **Confidence:** {low|medium|high}

## What this means
3–5 sentences in plain English for a non-finance reader. NO jargon (no "P50",
no "DCF", no "MoS", no "ERP", no "convergence ratio"). Cover, in order: what
the company does (one sentence); what the thesis is betting on (one sentence);
what our model says about price in plain language ("roughly fairly priced",
"meaningfully cheap", "pricey relative to the math") (one sentence); what we'd
do (one sentence); and one thing the reader should watch over the next quarter
(one sentence). This section anchors the report for an amateur investor.

## Thesis statement
One paragraph. The current analytical view, written for an experienced reader.

## Bull case
3–5 bullets. Each bullet ≤ 20 words, with one citation.

## Bear case
3–5 bullets. Same shape as bull.

## Top risks
Numbered list. Each with severity (1–5) and a one-sentence explanation.

## Monte Carlo fair value
A short paragraph stating P10 / P50 / P90 and how those compare to current
price, with the discount rate used and convergence ratio. Then THREE scenario
bullets to make the distribution actionable:

- **Bull (P75–P90):** one sentence describing the world that produces upside.
- **Base (P25–P75):** one sentence describing the central case.
- **Bear (P10–P25):** one sentence describing the world that produces downside.

## Action recommendation
One paragraph. What changes (if any) to thesis or position size, with specific
thresholds where possible (size + condition + threshold).

## Watchlist
3–5 bullets of forward-looking events / signals to track before the next
drill-in. Each bullet names the upstream agent whose work it relates to in
parentheses, e.g. "(filings)", "(news)", "(fundamentals)". Examples:
"Q3 earnings call (Aug 2026) — listen for AI capex guidance (news)";
"TSM yield disclosure in next 10-Q — supply concentration check (filings)";
"Inventory turnover trend in next quarter (fundamentals)". Phase 1 Triage
will read this section to seed thesis-specific monitoring rules.

## Evidence
Bulleted list of every source cited above with URLs or filing accession numbers.
```

`pdf_export.py` reads this Markdown and produces a styled PDF with the color palette from `§13`.

## 12. Streamlit UI

`ui/app.py` layout:

- **Sidebar:** thesis dropdown (loads JSON from `/theses/`), ticker input, "Run drill-in" button. Below: a "Run scan" button (Phase 0 stand-in for Triage that simulates 3 alerts from a fixture file).
- **Main area, top:** Synthesis report (rendered Markdown).
- **Main area, middle:** Monte Carlo histogram (matplotlib, styled with palette).
- **Main area, bottom:** four collapsible expanders, one per worker agent, showing their structured output and cited evidence.
- **Footer:** "Download PDF" button.

Use `st.status()` with `expanded=True` while the graph runs, streaming each node's completion. Cache thesis JSONs and yfinance responses with `@st.cache_data` (TTL 24h for yfinance, no TTL for theses).

## 13. Color palette

Warm-neutral editorial system with a single sage accent. Used consistently across the Streamlit UI, the PDF report, and any matplotlib charts. **No per-thesis color coding** — theses are distinguished by typography (thesis name in heading) and content.

| Role | Name | Hex |
|---|---|---|
| Brand accent | Botanical sage | `#2D4F3A` |
| Hero / closing surface | Parchment | `#F4ECDC` |
| Card / body surface | White | `#FFFFFF` |
| Inner panel surface | Eggshell | `#FBF5E8` |
| Outer border | Taupe | `#E0D5C2` |
| Internal divider | Bone | `#EDE5D5` |
| All text | Ink | `#1A1611` |

For the Monte Carlo histogram: bars use parchment (`#F4ECDC`) fill with sage (`#2D4F3A`) edge. P50 line: sage (`#2D4F3A`). P10 and P90 lines: taupe (`#E0D5C2`). Current price line: ink (`#1A1611`, dashed).

For the PDF report: parchment (`#F4ECDC`) cover and closing page; white (`#FFFFFF`) body pages with bone (`#EDE5D5`) dividers between sections; sage (`#2D4F3A`) section headers; ink (`#1A1611`) body text.

For the Streamlit theme (`.streamlit/config.toml`):

```toml
[theme]
primaryColor = "#2D4F3A"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#FBF5E8"
textColor = "#1A1611"
font = "sans serif"
```

Page-level background = parchment (`#F4ECDC`) for hero / landing sections; white (`#FFFFFF`) for body / report area.

## 14. Build order with acceptance tests

Build in this order. Do not skip ahead. Each step has a concrete acceptance test you can run before moving on.

1. **Scaffolding (1–2h).** Create the repo structure from `§4`. Install dependencies. Write `utils/openrouter.py` with a `get_client()` factory.
   - **Test:** `python -c "from utils.openrouter import get_client; print(get_client().chat.completions.create(model='anthropic/claude-haiku-4.5', messages=[{'role':'user','content':'say hi'}], max_tokens=20).choices[0].message.content)"` returns text.

2. **Data layer (3–4h).** Implement `data/edgar.py`, `data/yfin.py`, `data/chroma.py`. **Kick off ChromaDB ingestion of the 11-ticker AI cake universe in the background while you do step 3 and 4.**
   - **Test:** `chroma.query("NVDA", "data center capex outlook", k=5)` returns 5 chunks with non-empty `text`.

3. **Thesis JSONs (1h).** Hand-write the three JSON files in `/theses/` matching `§7`.
   - **Test:** `pydantic` validation passes for all three files.

4. **LangGraph skeleton (2–3h).** Build the graph in `agents/__init__.py` with **all five agent nodes returning hardcoded stub dicts**. Wire fan-out and fan-in.
   - **Test:** `graph.invoke({"ticker": "NVDA", "thesis": {...}})` returns a state dict with all keys filled and prints stub messages.

5. **Worker agents (8–12h).** Replace stubs with real LLM calls (model resolved per agent via the corresponding `MODEL_*` env var). Build in this order: Fundamentals → Filings → News → Risk. Confirm each works in isolation before running through the graph.
   - **Test per agent:** `python -m agents.fundamentals NVDA` prints a populated dict with all required keys.

6. **Monte Carlo engine (2–3h).** Implement `utils/monte_carlo.py` per `§10`.
   - **Test:** `simulate(revenue_now=60e9, growth_mean=0.15, growth_std=0.05, margin_mean=0.30, margin_std=0.05, exit_multiple_mean=25, exit_multiple_std=5, shares_outstanding=24.5e9)` returns a dict with `p50` between 0 and 1000.

7. **Synthesis agent (3–4h).** Single LLM call (model resolved via `MODEL_SYNTHESIS`) producing Markdown per `§11`. Implement `pdf_export.py`.
   - **Test:** End-to-end graph run produces a Markdown report with all required sections, and the PDF exporter writes a valid file.

8. **Streamlit UI + demo polish (6–8h).** Build `ui/app.py` per `§12`. Pre-cache 2 demo runs (NVDA on AI cake, EME on Construction) to JSON so the demo never waits on the synthesis LLM. Write README with run instructions.
   - **Test:** `streamlit run ui/app.py` opens a working app; clicking "Run drill-in" on NVDA loads the cached run instantly.

## 15. Phase scope

### 15.1 In scope for Phase 0 (hackathon — drill-in + MC + Streamlit)

The features defined in §§ 1–14 above. Three hand-written thesis JSONs stand in for Discovery; a "Run scan" button loads a fixture file in place of continuous Triage; alerts surface inside Streamlit only; memory persists to local JSON files in `./data_cache/memory/`.

### 15.2 In scope for Phase 1 (personal MVP — Notion + Telegram + continuous Triage)

Built on top of a working Phase 0:

| Feature | Phase 1 implementation |
|---|---|
| Notion memory of record | `data/notion.py` — read thesis notes, write reports, write/update alerts. One-way: read notes, write reports/alerts. |
| Bidirectional Telegram bot | `data/telegram.py` + `agents/router.py`. Slash commands (`/drill`, `/analyze`, `/scan`, `/note`, `/thesis`, `/help`) plus an LLM-backed natural-language fallback (model resolved via `MODEL_ROUTER`) for free-text requests. |
| Ad-hoc industry analysis | `agents/adhoc_thesis.py` synthesizes a `Thesis` from a free-text topic on demand and runs the existing drill-in graph against the top 1–3 anchor tickers. Discovery-lite. |
| Continuous Triage | `agents/triage.py` (LLM-backed; model resolved via `MODEL_TRIAGE`) + `scripts/run_triage.py` + `launchd` plist. Polls EDGAR + Tavily across all theses and pushes alerts to Telegram + Notion. |
| Public reachability | Decided at Step 12: Cloudflare Tunnel (free, laptop-awake) vs. DigitalOcean droplet (`$6/mo`, always-on) vs. inline-only Telegram (no dashboard link). |

### 15.3 Out of scope (deferred to Phase 2+)

Do not build these in Phase 0 or Phase 1, even if they seem easy:

| Feature | Phase | Notes |
|---|---|---|
| Full Discovery agent (halo graph builder, persistent topic monitoring) | 2+ | Step 10's `/analyze` is the Discovery-lite stand-in for now |
| Bidirectional Notion sync (write → Notion → user edits → re-read) | 2+ | Phase 1 is one-way only |
| Pattern detection / cross-thesis multiplicity scoring | 3 | The 4-ticker overlap (VRT, CEG, ANET, PWR) is data only, no automated scoring |
| Threshold learning / backtesting | 3 | Material thresholds are hand-set in thesis JSON until then |

## 16. Conventions

### 16.1 Code style

- **Type hints required** on all public functions. Use `pydantic` models for any structured agent output.
- **Async-first** for I/O. Agents are `async def run(...)`. Use `asyncio.gather` for parallel sub-tasks within an agent (e.g., the three Filings RAG queries).
- **`pathlib.Path` always** for filesystem operations; never raw `str` paths or `os.path.join`.
- **Pydantic models for cross-module contracts.** Raw dicts only inside a single module / function.
- **No `from x import *`.** Explicit imports only.
- **Constants at module top** in `UPPER_SNAKE_CASE`.
- **One class per file** when the class is > 50 lines; multiple small dataclasses can share a file.
- **Style:** `ruff` + `black`. Line length 100. Sorted imports. Tool config lives in `pyproject.toml`.

### 16.2 Code minimalism

- **Minimalism.** Prefer the fewest files, fewest lines, fewest abstractions that work. If a file is < 50 lines and used in one place, inline it. Don't add helper modules for hypothetical reuse.
- **DRY by rule of three.** First occurrence: write inline. Second: tolerate the duplication. Third: factor into a function or module-level constant. Never pre-abstract.
- **No premature abstraction.** This is a personal tool. Two concrete implementations beat one abstract one.

### 16.3 Dependencies

- **`requirements.txt` is the dependency manifest.** Every new top-level import must be added in the same change. Pin only when needed (compatibility, security). `pyproject.toml` is kept *only* for tool config.

### 16.4 Logging and errors

- **Logging:** stdlib `logging` configured in `utils/__init__.py`. Each agent logs its model, token count, and elapsed time. No `print()` outside `ui/`.
- **Error handling:** wrap every external call (yfinance, Tavily, OpenRouter, EDGAR, Notion, Telegram) in `tenacity` retry with `wait_exponential` and 3 attempts. If all fail, return a partial state with an `errors` field — never crash the whole graph.

### 16.5 Testing rigor (component-level gate)

A step is not done until its test plan passes end to end:

- Each step in the build plan has a **Test plan** with multiple named checks: happy path, edge cases, failure paths, and a manual verification pass where applicable.
- `pytest` runs unit-style tests on every commit. Tests against external services (OpenRouter, EDGAR, yfinance, Tavily, Notion, Telegram) are gated behind `pytest -m integration` and required to pass before a step's gate.
- Failures are root-caused and fixed; never `xfail` or skip a failing test to advance.
- Post the full test output in chat after every step and wait for the user's explicit "proceed" before starting the next.

### 16.6 Secrets

- Never log API keys, never write them to disk outside `.env`. `.env` is gitignored. `.env.example` contains placeholder values only.

### 16.7 Observability (mission control)

Single-user single-box system. No managed observability (Sentry/DataDog/Grafana) — overkill. Two layers + four read surfaces:

- **LangSmith tracing** (auto-instruments every LLM call when `LANGSMITH_TRACING=true`). Free tier covers personal use. Project name `finaq` by default. Used for per-LLM-call latency, tokens, full prompt/response, replay.
- **Local SQLite at `data_cache/state.db`** — single source of truth for graph_runs, node_runs, alerts, triage_runs, errors, daily_cost. Every agent writes through `data/state.py.record_*` from inside `_safe_node`. Schema migration is idempotent. Backed up with the rest of `data_cache/` on the droplet.

Read surfaces (each unlocked by an existing build step):
- **Streamlit Mission Control page** (Step 8, `ui/pages/mission_control.py`) — visual dashboard.
- **`/status` Telegram command** (Step 10) — text summary from your phone.
- **Healthchecks.io ping** (Step 11) — liveness alert when Triage stops firing.
- **systemd journalctl** (Step 12, droplet) — SSH-time deep dive.

Logging via stdlib `logging` is for *operational* output (what just happened); telemetry via `data/state.py` is for *historical / aggregate* queries. Don't conflate the two.

---

**Last updated:** see git log. Update this file before adding any new architectural decision so future Claude Code sessions stay on-spec.
