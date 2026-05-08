"""Autonomous CIO layer (Step 11).

The CIO sits *above* the LangGraph drill-in pipeline as a meta-agent. It runs:
  - On a heartbeat (twice daily, via launchd / systemd), sweeping every
    curated thesis ticker.
  - On demand from Telegram (`/cio`, `/cio TICKER`, `/cio TICKER thesis`).
  - On a catch-up cycle at boot if the last successful heartbeat was >8h ago.

For each `(ticker, thesis)` candidate the CIO decides one of:
  - **drill**   — invoke the existing graph (`agents.build_graph()`) for a fresh report.
  - **reuse**   — surface the most recent drill-in with a "still applies" qualifier.
  - **dismiss** — skip; nothing material has changed.

Module map:
  - `cio.memory`     — domain layer over `data.state` + `data.notion` (cooldown,
                       recent actions, user notes).
  - `cio.rag`        — RAG over the `synthesis_reports` ChromaDB collection
                       (past drill-ins, indexed by `scripts/index_existing_reports`).
  - `cio.planner`    — gates + LLM `decide()` per pair → `CIODecision`.
  - `cio.cio`        — orchestrator: sweep candidates, propose plan, execute,
                       notify. (Step 11.9)
  - `cio.notify`     — exec-summary composer + Telegram/Notion sends. (Step 11.10)
  - `cio.dispatcher` — cron CLI + Telegram /cio handler entry. (Step 11.10)

Public surface — kept thin to avoid pulling LLM / Chroma at import time.
Submodule references are deferred via `from cio.<mod> import <symbol>` inside
the consumer.
"""

from __future__ import annotations
