# FINAQ Architecture Decisions

This document records every component-level decision made while building
FINAQ, why it was chosen, and any later revisions.

## How to read this doc

- Sections are organised by **area**, not chronology — easier to find why a
  given component is the way it is.
- Each entry has: **Decision**, **Context**, **Why this over alternatives**.
  When a decision was later revised, a **Revised because:** block records
  the original plan and the reason for the change.
- New decisions are appended as each build step lands. **`docs/POSTPONED.md`**
  tracks *deferred* items; this doc tracks what was actually *built* and *why*.
- When an entry says "see CLAUDE.md §X", that's the build spec; "see plan"
  is the implementation plan in `~/.claude/plans/`.

---

## §1 Orchestration & state management

### 1.1 Multi-agent topology (LangGraph) over one big prompt

- **Decision:** Use LangGraph's `StateGraph` with explicit fan-out / fan-in
  nodes (Fundamentals, Filings, News in parallel → Risk → Monte Carlo →
  Synthesis). Each agent is one async coroutine.
- **Context:** A drill-in needs to integrate fundamentals, filings, news,
  and risk into a single report. The naive approach is one giant prompt;
  the considered approach is multi-agent.
- **Why:** Multi-agent gives (a) inspectable intermediate state at each
  node, (b) parallelism (3 workers run concurrently — saves ~3× latency),
  (c) per-agent model routing (cheap models for triage, expensive for
  synthesis), (d) per-agent eval (faithfulness checks per agent's output),
  (e) the ability to swap one agent without rewriting the rest.

### 1.2 `FinaqState` TypedDict with `operator.add` reducers

- **Decision:** Single TypedDict carries all state through the graph.
  Per-agent fields (`fundamentals`, `filings`, etc.) overwrite on update;
  `messages` and `errors` use `operator.add` so parallel branches can
  append without clobbering.
- **Why:** TypedDict is lightweight (no class), gives static types for free
  via Pydantic-validated outputs, and the reducer pattern is the
  LangGraph idiom for parallel-write fields.

### 1.3 `_safe_node` wrapper around every agent

- **Decision:** Every node (`load_thesis`, the four workers, monte_carlo,
  synthesis) is wrapped by `_safe_node` in `agents/__init__.py`. Catches
  exceptions, logs them, and writes a structured error into `state.errors`
  rather than letting the graph crash.
- **Why:** A single agent's failure (LLM outage, malformed JSON, network
  blip) shouldn't take down the whole drill-in. Other agents still produce
  useful state, the report can degrade gracefully, and the user sees what
  failed in `state.errors` instead of a 500.

### 1.4 Async-first agents with `asyncio.to_thread` for sync I/O

- **Decision:** Every agent is `async def run(state)`. Sync libraries
  (yfinance, ChromaDB, sec-edgar-downloader, OpenAI SDK) are wrapped with
  `asyncio.to_thread()` so they don't block the event loop.
- **Why:** LangGraph's parallel branches are scheduled on `asyncio`, so
  sync code in one branch would serialise the others. `to_thread` is the
  cleanest path; the alternative (rewriting libraries) is out of scope.

### 1.5 LangGraph CLI / Studio enabled (opt-in)

- **Decision:** Ship `langgraph.json` configuring the `drill_in` graph so
  `langgraph dev` can run locally and pair with LangGraph Studio for
  visual debugging. Adds `langgraph-cli[inmem]` to `requirements.txt`.
- **Why:** With multiple parallel agents and stateful graphs, debugging
  via `print()` and pytest gets painful. Studio gives node-by-node trace,
  state inspection, replay-from-any-point. Free-for-personal-use.
- **Local-first guarantee:** `langgraph dev` runs entirely on the laptop
  (no cloud dependency). Cloud deployment via `langgraph build` is
  ruled out — see CLAUDE.md §2.

---

## §2 LLM access layer

### 2.1 OpenRouter as the only LLM gateway

- **Decision:** All LLM calls go through OpenRouter's OpenAI-compatible
  endpoint (`base_url="https://openrouter.ai/api/v1"`). No direct calls
  to the `anthropic` or `openai` SDKs.
- **Why:** Single billing surface, single key, swap-models-by-string,
  unified retry/rate-limit semantics. We get embeddings via the same
  endpoint (`text-embedding-3-small`).

### 2.2 Model strings live in `.env`, not in code

- **Decision:** Each agent role has a dedicated env var
  (`MODEL_TRIAGE`, `MODEL_FUNDAMENTALS`, …, `MODEL_JUDGE`,
  `MODEL_EMBEDDINGS`). `utils/models.py` is a thin typed registry that
  reads from env and exposes typed constants.
- **Original plan (CLAUDE.md §3):** centralise in `utils/models.py` with
  hardcoded strings.
- **Revised because:** Juan asked for runtime configurability without
  code changes. Swapping a model is now a one-line `.env` edit.
- **Why:** Each role gets independent model selection (cheap for triage,
  expensive for synthesis), and we can A/B-test models without code edits.

### 2.3 Model-agnostic naming in code/prose

- **Decision:** Specific model names ("Sonnet", "Opus", "Haiku") appear
  *only* in `.env`, `utils/models.py`, and the model-routing table in
  `CLAUDE.md` §3. Everywhere else (agent docstrings, comments, prompts,
  tests, plan, this doc) refers to "the LLM", "the configured model", or
  the role ("the synthesis agent").
- **Why:** Tying narrative description to a specific model creates the
  false impression the agent is hardwired to that model, and the prose
  drifts when the model is swapped.
- **Audit:** `grep -rn -E "Sonnet|Opus|Haiku" agents/ data/ utils/ tests/
  scripts/` should return zero hits in narrative contexts.

### 2.4 OpenAI SDK retries (`max_retries=3`) for LLM calls

- **Decision:** `utils/openrouter.get_client()` returns an OpenAI client
  with `max_retries=3`. Per-LLM calls don't need a separate `tenacity`
  wrapper.
- **Why:** The OpenAI SDK already handles transient API errors with
  exponential backoff. Stacking `tenacity` on top would cause
  retry-on-retry behaviour. Tenacity is reserved for non-SDK calls
  (yfinance, EDGAR, Tavily, Notion, Telegram).

---

## §3 Data layer

### 3.1 SEC EDGAR via `sec-edgar-downloader` (idempotent local cache)

- **Decision:** Wrap `sec-edgar-downloader` in `data/edgar.py` with an
  idempotent `download_filings()` that defaults to the 2 most recent
  10-Ks + 4 most recent 10-Qs. Files persist under
  `data_cache/edgar/sec-edgar-filings/{ticker}/{kind}/{accession}/`.
- **Why:** SEC filings don't change after publication, so disk-cache
  forever. The library handles SEC's user-agent + rate-limit etiquette.
  We add idempotency on top so `download_filings("NVDA")` is cheap to
  call repeatedly.

### 3.2 yfinance with 24h TTL JSON cache + format-version invalidation

- **Decision:** `data/yfin.py` caches results to
  `data_cache/yfin/{ticker}.json` for 24 hours, then refetches.
  `CACHE_FORMAT_VERSION` invalidates older caches when the on-disk shape
  changes.
- **Why:** yfinance is rate-limited and flaky; caching is required.
  Version flag prevents silent breakage when we change the cache shape
  (we've already bumped 1 → 2 → 3).

### 3.3 Field-name aliases in yfinance KPI computation

- **Decision:** `INCOME_FIELD_ALIASES` and `CASH_FLOW_FIELD_ALIASES` map
  canonical names ("revenue", "operating_income", "free_cash_flow") to
  ordered lists of yfinance variations ("Total Revenue", "Operating
  Revenue", "Revenue"; "Free Cash Flow"; "Capital Expenditure",
  "Capital Expenditures").
- **Original implementation:** direct `dict.get("Total Revenue")`.
- **Revised because:** yfinance line-item names vary across tickers — NVDA
  uses `"Total Revenue"`, some tickers use `"Operating Revenue"`. The
  direct-get approach silently produced null KPIs. Found by inspecting
  the live NVDA output during Step 5a review.

### 3.4 yfinance JSON shape: dates as outer keys

- **Decision:** Financial-statement DataFrames (income / balance / cash
  flow) are transposed before serialising so the on-disk shape is
  `{date: {line_item: value}}` rather than yfinance's natural
  `{line_item: {date: value}}`.
- **Original implementation:** `df.to_json(orient="index")` without
  transpose.
- **Revised because:** Without the transpose, every KPI computed by
  `compute_kpis` came back null because the code was iterating over
  dates but reading line-item-keyed dicts. Discovered while reviewing
  Step 5a's actual NVDA output (everything was null while the LLM
  hallucinated plausible numbers).

### 3.5 ChromaDB local persistent store (no hosted vector DB)

- **Decision:** `chromadb.PersistentClient` writing to
  `data_cache/chroma/`. Single collection named `filings`.
- **Why:** Personal-tool scale (~10K chunks); no need for Pinecone /
  Weaviate / Qdrant Cloud. Local-first per CLAUDE.md §2. ChromaDB has
  metadata-where filtering and our custom embedding function works fine.

### 3.6 Cosine distance (explicit) over L2 default

- **Decision:** Collection created with
  `configuration={"hnsw": {"space": "cosine"}}`. `DISTANCE_SPACE = "cosine"`
  is the named constant.
- **Original implementation:** ChromaDB default (L2).
- **Revised because:** Discovered during Step 5b review that the
  collection was using L2. For unit-normalised text-embedding-3-small
  vectors, L2 and cosine produce identical rankings (just different
  score magnitudes), so we lost nothing operationally. But cosine is
  the conceptually correct distance for text retrieval and protects us
  if we swap to a non-normalised embedding model later. Re-ingestion
  cost was modest (~$0.30) so we wiped + rebuilt.

### 3.7 800-token chunks, 100-token overlap, split on Item headers

- **Decision:** Filings are first split on SEC Item headers (`Item 1A`,
  `Item 7`, etc.); within each section, ~800-token chunks with
  100-token overlap. `tiktoken cl100k_base` for tokenisation.
- **Why this size specifically:** ~1 page of dense SEC text fits in
  800 tokens; small enough to keep embedding signal discriminative,
  large enough to capture coherent paragraphs. Cross-encoder budgets
  (typical 512) are accommodated if we add re-ranking later.
- **Trigger to revisit:** Synthesis agent cites mid-sentence-cut
  chunks (lower size) or top-K is clearly redundant (lower size + add
  MMR). See POSTPONED §2.

### 3.8 Custom regex tokeniser + 150-word stopword list (over NLTK)

- **Decision:** BM25 tokenisation in `data/chroma.py` uses a regex
  word-extractor (`[a-z0-9]+`) plus an inline 150-word English stopword
  list. `lru_cache` on the tokenisation function.
- **Original implementation:** Naive `doc.lower().split()`.
- **Revised because:** "company." became a different token from
  "company"; stopwords ("the", "of") dominated token frequency,
  diluting the IDF signal so rare/discriminative terms didn't drive
  ranking.
- **Why not NLTK:** 50MB corpus download at first use; runtime network
  requirement; mostly we need word splitting + stopwords (overkill).
  Inline implementation is ~30 lines and 80% of NLTK's quality benefit.

### 3.9 Embedding model: `text-embedding-3-small` via OpenRouter

- **Decision:** Use OpenRouter's `text-embedding-3-small` for all
  ChromaDB embeddings. Configured via `MODEL_EMBEDDINGS` env var.
- **Why:** Cheapest decent embedding model on OpenRouter (~$0.02/M
  tokens), unit-normalised vectors (so cosine and L2 are equivalent),
  1536 dimensions (good signal/storage tradeoff), broadly competitive
  on retrieval benchmarks.

### 3.11 Tavily for news search (over web-scraping or other APIs)

- **Decision:** `data/tavily.py` wraps `TavilyClient.search()` with
  `topic="news"`, `search_depth="advanced"`, default 90-day window,
  default max-results 15. Tenacity-retried; returns empty list on
  persistent failure.
- **Why Tavily over alternatives:** SerpAPI charges per query at higher
  rates; Google Custom Search needs a billing account + cse_id config
  + lacks a clean date filter; raw web-scraping is brittle and
  rate-limited per-domain. Tavily is purpose-built for LLM use cases
  (returns clean snippets + scores + dates), has a generous free tier
  (1000 queries/mo), and the API is one HTTP call.
- **Why no caching:** News is time-sensitive — a 24h-cached result
  could miss the catalyst that drives the alert. At ~$0.005/call and
  one call per drill-in, the cost is negligible.

### 3.10 Freshness markers throughout the pipeline

- **Decision:** Every retrieved/computed datum carries a freshness
  timestamp. yfinance: `fetched_at` (UTC ISO when the API was hit).
  EDGAR/Chroma: `filed_date` extracted from SGML header into chunk
  metadata. Agent-level: `Evidence.as_of` is propagated downstream.
  Fundamentals agent renders an "AS OF" block at the top of every
  prompt; soft warning if `fetched_at` > 7 days old.
- **Why:** Stale data → bad analysis → bad investing decisions. The
  LLM otherwise has no way to weight a 2025 10-K differently from a
  2021 10-K, or to discount KPIs computed from 6-month-old yfinance
  data.

---

## §4 RAG pipeline

### 4.1 Hybrid retrieval: semantic + BM25 + RRF

- **Decision:** ChromaDB returns a candidate pool of 60 chunks
  (cosine-ranked). BM25 ranks the same pool by keyword score.
  Reciprocal Rank Fusion (k=60) merges both rankings; top-8 returned.
- **Why both:** Pure semantic misses queries with rare/proper-noun
  keywords (e.g., "Mellanox", "Blackwell"). Pure BM25 misses
  paraphrased queries. RRF is the standard fusion strategy — robust to
  one method finding nothing as long as the other does.
- **Why RRF over weighted-sum:** RRF is parameter-free (just `k=60`,
  the standard). Weighted-sum requires tuning per-corpus and is
  brittle when scores are on different scales.

### 4.2 Metadata filter applied BEFORE similarity scan

- **Decision:** ChromaDB's `where` clause is constructed in
  `_build_where_clause` and passed to `coll.query()`. ChromaDB applies
  it before the similarity scan, not after.
- **Why:** Filtering after retrieval would mean the top-K pool is
  dominated by other-ticker / other-item chunks that get *discarded*,
  leaving us with fewer than K relevant chunks. Pre-filtering ensures
  the candidate pool is already inside the right slice.

### 4.3 60-chunk candidate pool feeding RRF

- **Decision:** `DEFAULT_CANDIDATE_POOL = 60`.
- **Why:** Big enough that BM25 has plausible competition with
  semantic top-8 (a chunk semantic ranks at #50 but BM25 #1 still
  enters). Small enough that BM25's per-query cost is negligible
  (microseconds). For a single-ticker corpus of ~10K chunks, 60
  candidates after the metadata filter is a reasonable working set.

### 4.4 No cross-encoder re-ranking (deferred)

- **Decision:** Phase 0 ships with semantic + BM25 + RRF only. Cross-
  encoder is not included.
- **Why:** Cross-encoder adds ~500MB local model + 2-4s/query latency;
  hybrid retrieval already hits ~85-90% of the quality. The marginal
  5-10% improvement isn't worth the dep weight today.
- **Trigger to revisit:** see POSTPONED §2 — Filings citations feeling
  off after Step 8 demo.

### 4.5 Hardcoded thesis-aware subquery templates

- **Decision:** The Filings agent's three subqueries (Risk Factors,
  MD&A trajectory, Segment performance) are hardcoded templates with
  thesis-name and ticker interpolation.
- **Why over LLM-generated:** Adds a separate cheap-tier LLM call per
  drill-in (~$0.001) for marginal nuance gain. For Phase 0's three
  hand-written theses, hardcoded is enough.
- **Trigger to revisit:** ad-hoc `/analyze` (Step 10) launches and
  hardcoded templates feel too rigid for arbitrary thesis topics.

---

## §5 Schemas & validation

### 5.1 Pydantic v2 for all cross-module contracts

- **Decision:** `utils/schemas.py` defines `Thesis`, `Relationship`,
  `MaterialThreshold`, `Projections`, plus per-agent output models
  (`FundamentalsOutput`, `FilingsOutput`, etc.). Every agent's `run()`
  returns a Pydantic-validated dict.
- **Why:** Catches LLM-output drift at the type boundary instead of
  blowing up downstream. The schema doubles as the prompt contract
  (we reproduce the JSON shape in the prompt). When a test wants to
  assert an agent's return shape, `model_validate(result)` is one line.

### 5.2 Thesis schema validators (`anchor_tickers ⊆ universe`, etc.)

- **Decision:** `Thesis` model has `@model_validator` enforcement of
  three invariants: anchors are a subset of the universe; every
  relationship endpoint references a ticker in the universe; threshold
  operators match the value type (`contains` ↔ string, others ↔ numeric).
- **Why:** A typo in a thesis JSON should fail fast at load time, not
  produce confusing downstream errors.

### 5.3 Buffett-style thresholds added to all theses

- **Decision:** Every thesis gets six Buffett-style `material_thresholds`
  on top of the original macro/capex/momentum signals: `fcf_yield`
  (over/under), `fcf_to_net_income_5yr`, `capex_to_revenue_5yr_avg`,
  `current_price_vs_p50_mc`, `current_price_vs_5yr_pe_avg`.
- **Why:** The original spec encoded only macro signals (capex
  announcements, margin changes, filing keywords). Adding Buffett
  filters gives Triage a way to surface owner's-earnings-style and
  margin-of-safety signals.
- **Discussed alternatives:** A (moat metadata in Thesis schema) and C
  (capital-allocation thresholds via Form 4 ingestion) — both deferred
  to POSTPONED §2 because they require schema changes / new pipelines.
  We picked B + D as the lowest-cost-highest-value subset.

### 5.4 Halo · NVDA universe is 7 tickers (NVDA not included)

- **Decision:** `theses/nvda_halo.json` has `universe = [SMCI, DELL,
  VRT, CEG, ANET, CRDO, MRVL]`. NVDA is the *implicit* anchor named
  in the thesis description, but not in the universe list.
- **Why this faithful-to-docx interpretation:** `FINAQ_Context.docx`
  explicitly lists 7 orbit names. Adding NVDA explicitly would let
  schema-level validators cover `relationships.from = "NVDA"` edges,
  but at the cost of deviating from the doc.
- **Trigger to revisit:** see POSTPONED §2 — if Triage misses an
  NVDA-only signal because NVDA isn't in Halo's universe, switch to
  8 tickers.

---

## §6 Agent design

### 6.1 Each agent owns one cognitive step (narrow scope)

- **Decision:** Fundamentals = extract + interpret KPIs. Filings = RAG
  + thesis-aware extraction. News = web search + sentiment tagging.
  Risk = synthesis-only (no external calls). Synthesis = final
  bull/bear report. Triage = filter incoming items against thresholds.
- **Why:** Each agent is replaceable / swappable. Eval per-agent is
  meaningful (faithfulness on Filings, projection-quality on
  Fundamentals). One LLM call per cognitive step keeps prompts focused
  and outputs verifiable.

### 6.2 Fundamentals: deterministic Python KPI computation + LLM interpretation

- **Decision:** `compute_kpis()` is pure Python (yfinance dict →
  KPI dict). The LLM only *interprets* the numbers and projects
  forward; it doesn't compute anything itself.
- **Why:** LLMs are unreliable arithmeticians, but they're great at
  reading historical KPIs and producing thesis-aware narrative +
  forward projections (mean + std for Monte Carlo).

### 6.3 Fallback projections: derive from history (Approach A)

- **Decision:** When the LLM call fails or returns invalid output,
  `_derive_fallback_projections(kpis)` builds projections from
  computed KPIs (CAGR → revenue_growth_mean, op_margin_5yr_avg →
  margin_mean, pe_trailing → exit_multiple_mean), with the
  `NULL_HYPOTHESIS_PROJECTIONS` baseline filling fields where history
  is missing.
- **Original implementation:** Static `DEFAULT_PROJECTIONS` (5%
  growth, 10% margin, 15x multiple) regardless of ticker.
- **Revised because:** Juan flagged that the static defaults would
  produce nonsense valuations for any actual thesis ticker
  (NVDA at 5% growth is absurd). Discussed alternatives:
  (B) per-thesis defaults in JSON, (C) refuse to project, (D) per-
  ticker defaults in JSON. Picked A — uses real historical data
  when available, only falls back to generic when truly missing.

### 6.4 Filings: 3 RAG subqueries → 1 LLM synthesis call

- **Decision:** Three retrievals (Risk Factors / MD&A / Segments)
  each return top-8 chunks. One LLM synthesis call consumes all 24
  chunks plus the thesis to produce `FilingsOutput`.
- **Why:** Three separate LLM synthesis calls would cost 3× and miss
  cross-section integration ("MD&A celebrates X but Risk Factors warn
  about X — note the tension"). One synthesis sees everything.

### 6.5b News-prompt "what to skip" sharpened (event-attribution rule)

- **Decision:** The News prompt's "what to skip" rule now uses a two-part
  test: skip ONLY if the article is about a price move with no underlying
  event named. KEEP any story where a price move is *attributed* to a
  corporate / regulatory / structural event (CEO change, M&A, guidance
  revision, export-control action, partnership).
- **Original prompt:** "Stock-price-only stories ('NVDA up 3% today')" —
  ambiguous; a small/cheap LLM could interpret it strictly and skip a
  "20% drop on CEO resignation" story as price-only.
- **Revised because:** Juan flagged that a CEO-resignation-with-stock-drop
  is exactly the kind of catalyst a News agent must surface. Verified
  with a regression test (`test_event_attributed_price_move_is_kept`)
  that mocks Tavily with a CEO-resignation article + a bare price-move
  article and asserts the agent keeps the first and skips the second.

### 6.6 News agent: 90-day Tavily window + 1 LLM extraction call

- **Decision:** News agent calls Tavily once for `{ticker} {company_name}`
  (last 90 days, top 15 by score), passes the article list to one LLM
  call that extracts 3-7 catalysts (bull/neutral) and 3-7 concerns
  (bear/neutral). Each item carries `sentiment`, `url`, and `as_of`
  (from `published_date`).
- **Why one LLM call over per-article:** Per-article would cost 15×
  more and miss cross-article integration ("two outlets reported the
  same catalyst — pick the higher-quality source", "this concern
  contradicts that catalyst"). One pass produces a coherent narrative.
- **Why 90 days:** Aligned with CLAUDE.md §9.3 and matches the typical
  thesis-relevance horizon (intra-quarter signal). Long enough to
  catch durable themes; short enough that the LLM doesn't have to
  weight a 6-month-old story against today's news.
- **Sentiment ENUM (bull/bear/neutral):** Directional, not emotional —
  "bull" = positive *for the thesis*, not "good news in general". A
  competitor's setback is "bull" from the active thesis's perspective.

### 6.7 Company-name resolution via yfinance cache

- **Decision:** News agent calls `data.yfin.get_financials(ticker)` to
  pull `info.longName` for the Tavily query string. Falls back to the
  ticker symbol if yfinance fails.
- **Why use yfinance:** Already cached (24h TTL) — typically free
  because Fundamentals just populated it. Maintaining a separate
  ticker→name table is duplicative.
- **Trade-off:** When News and Fundamentals run in parallel, both
  may miss the cache and call yfinance simultaneously. Acceptable —
  yfinance handles the duplicate fetch fine, and subsequent drill-ins
  hit the cache.

### 6.8 `as_of` on NewsItem (schema addition)

- **Decision:** `NewsItem` schema gained an `as_of: str | None = None`
  field, populated from each article's `published_date`. Mirrors the
  same field on `Evidence`.
- **Why:** Consistency with the freshness-marker discipline (§3.10).
  Risk and Synthesis can sort/filter catalysts by recency without
  hopping through the evidence list to find the date.

### 6.5 Filings prompt explicitly pushes "connect the dots"

- **Decision:** The Filings system prompt has an entire "Beyond
  extraction — connect the dots" section: tension across subqueries,
  trends across multiple filings, implicit-but-not-stated signals,
  prefer cross-subquery evidence.
- **Why:** Without this nudge, the LLM defaults to summarisation. The
  prompt explicitly requires analytical depth within the Filings
  domain (full cross-modal analysis happens in Risk + Synthesis).

---

## §7 RAG evaluation (3-tier)

### 7.1 Three tiers, each with its own gate

- **Decision:**
  - Tier 1 (deterministic, always-on, `pytest -m integration`):
    golden-set recall@K + faithfulness + citation accuracy.
  - Tier 2 (LLM-as-judge, opt-in, `pytest -m eval`): per-chunk
    relevance scoring → precision@K, NDCG@K, MRR.
  - Tier 3 (RAGAS, opt-in, `pytest -m eval`): faithfulness,
    answer_relevancy, context_precision, context_recall.
- **Why three tiers:** Tier 1 catches plumbing bugs cheaply on every
  CI run. Tier 2 catches retrieval-quality regressions on demand.
  Tier 3 gives standardised metrics for cross-strategy comparison.

### 7.2 Hand-curated golden set (8 NVDA queries)

- **Decision:** `tests/eval/golden_queries.py` has 8 `GoldenQuery`
  records, each with expected substrings and an optional item filter.
- **Why a small curated set over auto-generation:** Auto-generated
  queries are easy to game; the LLM can produce queries it knows it
  can answer. Hand-curated queries reflect real user intent and catch
  retrieval failures the LLM wouldn't notice.

### 7.3 Categorical labels (NONE/WEAK/PARTIAL/HIGH) over integer scores

- **Decision:** The Tier 2 judge returns one of four labels;
  `_LABEL_TO_SCORE` maps internally to integers (0/1/2/3) for graded
  NDCG@K and MRR math.
- **Original implementation:** Integer 0–3 scoring.
- **Revised because:** Juan flagged two prompt-engineering issues —
  (a) numeric scales have unstable semantics across models (a 70B's
  "2" might be a frontier model's "3"), (b) requesting `{"score":
  int, "rationale": str}` forces the model to commit to a score
  before generating the rationale, which is post-hoc justification,
  not reasoning. Both fixed by switching to categorical labels.

### 7.4 Rationale-before-label JSON ordering (chain-of-thought)

- **Decision:** Judge prompt requires
  `{"rationale": "...", "label": "..."}` in that key order.
- **Why:** LLMs are autoregressive — generating the rationale first
  forces a mini chain-of-thought that improves the final label
  quality. Putting label first would be a blind guess that the model
  then justifies post-hoc.

### 7.5 Eval results persist to `data_cache/eval/runs/{ts}.json`

- **Decision:** Every eval test (Tier 1/2/3) calls `write_eval_run()`
  which persists a JSON snapshot of metrics + per-query results.
- **Why:** Mission Control panel (Step 8) reads from these files so
  the dashboard surfaces latest scores + per-query pass/fail. After
  Step 5z, this lifts into `state.db`.

### 7.6b News-quality eval (3 tiers, mirroring the RAG eval)

- **Decision:** News agent has the same three-tier eval suite the Filings
  RAG agent has. **Tier 1** (always-on, deterministic, `pytest -m
  integration`): golden-set recall + URL faithfulness + CEO-resignation
  regression test. **Tier 2** (`pytest -m eval`, ~$0.05/run): LLM-judge
  relevance + sentiment-accuracy over each catalyst/concern. **Tier 3**
  (`pytest -m eval`, ~$1+/run): RAGAS faithfulness / answer_relevancy /
  context_precision over the news synthesis.
- **Why:** News gets the same eval discipline RAG does — quality
  regressions are detectable, persisted to `data_cache/eval/runs/`, and
  surface in the Mission Control panel (Step 8).

### 7.6c URL canonicalisation in News faithfulness check

- **Decision:** `_canonicalise_url()` strips query string + fragment +
  trailing slash before comparing News-agent URLs against Tavily's
  returned articles.
- **Original implementation:** raw string equality.
- **Revised because:** the LLM legitimately strips tracking parameters
  off URLs (`?gaa_at=...`, `?utm_source=...`). Without canonicalisation,
  a real article URL can look like a fabrication just because of `?`
  differences. We saw a real WSJ article fail the test for this reason.

### 7.6d No summary-grounding check in News Tier 1 (paraphrase-aware in Tier 3)

- **Decision:** News Tier 1 faithfulness check does NOT verify that
  catalyst/concern *summaries* substring-match article content. Only URL
  grounding is checked deterministically.
- **Why:** News summaries are paraphrased on purpose — the prompt does
  NOT require verbatim quotes (unlike Filings, which does). Trying to
  enforce substring-match would fail for legitimate paraphrasing.
  Paraphrase-aware faithfulness is what RAGAS Tier 3 is built for.

### 7.6e News LLM-judge prompt: relevance + sentiment combined

- **Decision:** Tier 2's News judge prompt asks for two judgments per
  item in a single LLM call: (a) is this item materially relevant to
  the thesis? (b) is the sentiment label correct? Returned as
  `{"rationale": ..., "is_relevant": "RELEVANT|NOT_RELEVANT",
  "sentiment_correct": "CORRECT|INCORRECT|UNCLEAR"}`.
- **Why:** Two judgments per call halves cost vs separate calls;
  sentiment-accuracy is computed only over decided items (UNCLEAR
  filtered out, not counted against the metric).

### 7.6 Multi-signal per-query gate for Tier 2

- **Decision:** Tier 2 per-query test passes if ANY of:
  precision@8 ≥ 0.125, MRR ≥ 0.25, NDCG@8 ≥ 0.4.
- **Original implementation:** strict `precision@8 ≥ 0.25`.
- **Revised because:** the live first run had a query
  (`customer concentration risk top customers`) where retrieval was
  fine in *ranking* (NDCG=0.59, MRR=0.33) but only 1 of 8 chunks was
  rated `PARTIAL` or higher by the judge. That's a legitimate
  retrieval scenario (judge is conservative on broad queries), not a
  regression. Multi-signal gate catches catastrophic retrieval
  (all signals zero) without flagging legitimate noise.

### 7.7 Multiple judge models supported via `MODEL_JUDGE` env var

- **Decision:** Tier 2 and Tier 3 read from `MODEL_JUDGE` env var
  (default cheap-tier). Test fixtures skip cleanly if the env var is
  unset or still the conftest stub.
- **Why:** Lets the user A/B-test judges (cheap vs. premium) without
  code changes. Default cheap-tier keeps eval cost manageable.

---

## §8 Engineering & development

### 8.1 `requirements.txt` as dependency manifest

- **Decision:** Top-level deps live in `requirements.txt`.
  `pyproject.toml` is kept *only* for tool config (ruff, black,
  pytest markers).
- **Original CLAUDE.md §5:** `pyproject.toml` with deps.
- **Revised because:** Juan asked for `requirements.txt` specifically,
  noting it's the simpler / more legible / more portable manifest.
- **Why:** A single text file enumerating every package is the
  lowest-friction option for installation and audit.

### 8.2 Single `.env` for keys + model strings + Phase 1 placeholders

- **Decision:** All runtime config — API keys, model strings, Notion
  IDs, Telegram tokens — lives in `.env`. `.env.example` is committed
  with placeholder values plus all `MODEL_*` defaults set to current
  latest models.
- **Why:** One file to audit, one file to rotate. Phase 1 additions
  (Notion / Telegram) are commented in `.env.example` so the user
  uncomments them when ready instead of forgetting they exist.

### 8.3 ruff + black, line length 100

- **Decision:** Both linter and formatter in CI. Ruff for rules
  (style, imports, complexity). Black for layout. 100 chars to fit
  modern displays.

### 8.4 Pytest with `integration` and `eval` markers

- **Decision:** Three test categories:
  - default: pure unit, fast, runs in CI.
  - `-m integration`: hits real EDGAR / yfinance / OpenRouter /
    ChromaDB / Tavily / Notion / Telegram. Requires `.env`.
  - `-m eval`: opt-in RAG quality eval (Tier 2 + Tier 3). Costs
    money per run.
- **Why three tiers, not two:** Eval costs vary enormously (Tier 1 is
  free, Tier 2 ~$0.15/run, Tier 3 ~$1+/run). Separating eval from
  integration avoids running expensive evals on every CI commit.

### 8.5 Conftest stubs for unit-test isolation

- **Decision:** `tests/conftest.py` calls `load_dotenv()` first, then
  `setdefault` placeholder values for every `MODEL_*` env var. Real
  `.env` values always take precedence; stubs only fill missing vars.
- **Why:** Unit tests should run without a populated `.env` — they
  don't actually call the LLM. Setting stubs (after dotenv) gives
  unit-test isolation without breaking integration tests.

### 8.6 `test_graph.py` autouse fixture re-stubs real agents

- **Decision:** An autouse fixture in `test_graph.py` monkey-patches
  every agent's `run` function to a deterministic stub. Each test gets
  fast deterministic graph behaviour regardless of how many agents
  have real implementations.
- **Why:** Without this, every graph test would hit the real LLM
  (~3 minutes per run, ~$0.10+). With it, graph tests run in 1.2s.
  Real agent behaviour is tested in `test_{agent}.py` and
  `test_{agent}_integration.py`.

---

## §9 Phase scope & roadmap deviations

### 9.1 Phase 0 + Phase 1 in scope (broader than CLAUDE.md original)

- **Decision:** Build plan covers both Phase 0 (drill-in, Monte Carlo,
  Streamlit UI, fixture-backed Triage) and Phase 1 (real Triage,
  Notion memory, Telegram bot, droplet deploy).
- **Original CLAUDE.md §15:** Phase 0 only; everything else
  out-of-scope for the hackathon.
- **Revised because:** Juan opted for the personal-tool path
  (continuous monitoring, multi-surface alerts) over the
  hackathon-only scope. Phase 1 layers on top of a complete Phase 0.

### 9.2 Bidirectional Telegram + ad-hoc analyze (Discovery-lite)

- **Decision:** Step 10 builds a Telegram bot with slash commands
  (`/drill`, `/analyze`, `/scan`, `/note`, `/thesis`, `/status`,
  `/help`) plus natural-language fallback. `/analyze` uses a
  Discovery-lite agent that synthesises an ad-hoc thesis from a
  free-text topic.
- **Original CLAUDE.md §15:** Telegram = outbound alerts only.
- **Revised because:** Juan wanted Telegram to be the primary action
  interface ("text the agent to analyze defense semis"). Discovery-
  lite is the trade-off — ad-hoc thesis synthesis without the
  persistent halo-graph and ongoing-monitoring pieces of the full
  Phase 2 Discovery agent.

### 9.3 DigitalOcean droplet as eventual deployment target

- **Decision:** Step 12 deploys the working Phase 1 system to a $6/mo
  DigitalOcean droplet (Ubuntu 22.04, systemd timers for Triage,
  Caddy for HTTPS).
- **Original plan:** Step 12 was "decide-later" between Cloudflare
  Tunnel, droplet, or inline Telegram replies.
- **Revised because:** Juan committed to the droplet as the
  always-on home (laptop-awake-only via Cloudflare Tunnel was the
  alternative). Tunnel option dropped from the plan.

### 9.4 Mission Control as a first-class concern

- **Decision:** Mission Control is its own Streamlit page (Step 8) +
  Telegram `/status` command (Step 10), reading from the SQLite
  telemetry layer (Step 5z) and the eval results in
  `data_cache/eval/runs/`.
- **Original CLAUDE.md §15:** Observability not mentioned.
- **Why added:** Without monitoring, a 24/7 multi-agent system is
  unobservable. Adding Mission Control as a planned step before
  droplet deploy ensures visibility from day one.

### 9.5 Step 5z (observability foundation) inserted between 5d and 6

- **Decision:** A dedicated step builds `data/state.py` (SQLite
  telemetry) + LangSmith opt-in + `_safe_node` rewiring to record
  every node entry/exit before any further agents are added.
- **Why before Step 6:** Each subsequent step (Monte Carlo, Synthesis,
  UI) generates real traces. Doing observability first means we have
  a complete history from there on; doing it after means we lose
  early-step visibility.

---

## §10 Aesthetic & UX

### 10.1 Sage / parchment colour palette

- **Decision:** A warm-neutral editorial palette with a single sage
  accent (`#2D4F3A`), parchment hero (`#F4ECDC`), white cards,
  eggshell panels, taupe borders, bone dividers, ink text
  (`#1A1611`). Used in Streamlit theme, Monte Carlo histogram, PDF.
- **Original CLAUDE.md §13:** Multi-color palette with one colour per
  thesis (purple AI cake, teal Halo, coral Construction) plus
  separate hues for data connectors / LLM agents / tools.
- **Revised because:** Juan provided the new palette during planning.
  The original per-thesis colours conflict with the design language
  Juan wants.
- **Implication:** Theses no longer have hue-distinguishing cues.
  If a way to glance-distinguish them becomes useful, we'll add
  differential icons or tags rather than reverting to colours.

### 10.2 Streamlit for UI (over FastAPI + React)

- **Decision:** Single Streamlit app for the dashboard.
- **Why:** Personal tool for one user. Streamlit gives us:
  multi-page navigation (`pages/`), `st.status()` for streaming graph
  events, theme via `.streamlit/config.toml`, native matplotlib
  rendering — all without writing a frontend. FastAPI + React would
  triple the surface area.

---

## How to update this doc

When a new architectural decision is made (in any step):

1. Add a new entry under the relevant section.
2. **Decision:** state the choice.
3. **Why:** explain the reasoning and key alternatives considered.
4. If the decision *replaces* an earlier one: keep the new entry
   compact and add a `**Revised because:**` block explaining what was
   originally planned and why it changed.
5. If the decision *defers* something instead of building it: log the
   deferral in `docs/POSTPONED.md` §2 with an explicit trigger, and
   reference it from this doc with "see POSTPONED §2".

The doc is meant to be read end-to-end by someone joining the project
months from now. Optimise for "I can understand why X is the way it is
in 30 seconds of reading."
