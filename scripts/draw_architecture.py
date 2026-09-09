"""Generate the FINAQ Fabric Showcase Hybrid architecture diagram.

Produces docs/architecture_fabric.png. Re-run after any architecture change so
the PNG stays in sync with the implementation plan and ARCHITECTURE.md §10.

Three horizontal swimlanes (External / Microsoft Fabric / Plain Azure) with
labelled component boxes and arrows for the key dataflows. Palette per
CLAUDE.md §13 (sage / parchment / ink). Existing resources get a dashed
border; new (Bicep-provisioned) resources get a solid border.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SAGE = "#2D4F3A"
PARCHMENT = "#F4ECDC"
WHITE = "#FFFFFF"
EGGSHELL = "#FBF5E8"
TAUPE = "#E0D5C2"
BONE = "#EDE5D5"
INK = "#1A1611"

LANE_EXT = "#F8F0DC"
LANE_FABRIC = "#F1F4EF"
LANE_AZURE = PARCHMENT

OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "architecture_fabric.png"


def add_box(ax, x, y, w, h, title, subtitle, *, fill, edge=SAGE,
            existing=False, fs_title=8.5, fs_sub=6.5):
    linestyle = (0, (4, 2)) if existing else "solid"
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fill, edgecolor=edge, linewidth=1.3, linestyle=linestyle,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h - 0.16, title,
        ha="center", va="top",
        fontsize=fs_title, fontweight="bold", color=INK,
    )
    ax.text(
        x + w / 2, y + h / 2 - 0.12, subtitle,
        ha="center", va="center",
        fontsize=fs_sub, color=INK, wrap=True,
    )


def add_arrow(ax, src, dst, *, curve=0.0, color=SAGE, lw=1.0, label=None):
    arrow = FancyArrowPatch(
        src, dst,
        arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (src[0] + dst[0]) / 2
        my = (src[1] + dst[1]) / 2 + 0.05
        ax.text(
            mx, my, label, fontsize=6.5, color=SAGE, ha="center",
            fontweight="bold",
            bbox=dict(facecolor=WHITE, edgecolor="none", pad=2, alpha=0.92),
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    fig.patch.set_facecolor(WHITE)

    ax.text(8.0, 9.75, "FINAQ — Microsoft Fabric Showcase Hybrid",
            ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(8.0, 9.4,
            "End-state deployment · branch fabric-foundry · rg-finaq-prod · East US 2",
            ha="center", va="center", fontsize=9, color=INK, style="italic")

    ax.add_patch(FancyBboxPatch((0.2, 8.0), 15.6, 1.15,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=LANE_EXT, edgecolor=TAUPE, linewidth=0.8))
    ax.add_patch(FancyBboxPatch((0.2, 4.3), 15.6, 3.55,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=LANE_FABRIC, edgecolor=TAUPE, linewidth=0.8))
    ax.add_patch(FancyBboxPatch((0.2, 0.7), 15.6, 3.45,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=LANE_AZURE, edgecolor=TAUPE, linewidth=0.8))

    ax.text(0.4, 9.0, "External services", fontsize=11, fontweight="bold", color=SAGE)
    ax.text(0.4, 7.7, "Microsoft Fabric — workspace finaq",
            fontsize=11, fontweight="bold", color=SAGE)
    ax.text(0.4, 4.0, "Plain Azure — rg-finaq-prod (East US 2)",
            fontsize=11, fontweight="bold", color=SAGE)

    ext_y, ext_h = 8.12, 0.78
    ext = [
        ("SEC EDGAR", "10-K / 10-Q / 6-K / 20-F"),
        ("yfinance", "Prices, statements (24h TTL)"),
        ("Finnhub", "Company news (90d window)"),
        ("Notion", "Theses notes ⇄ reports/alerts"),
        ("Telegram", "Alerts + slash commands"),
        ("LangSmith", "LLM trace observability"),
    ]
    w, gap = 2.5, 0.1
    for i, (title, sub) in enumerate(ext):
        x = 0.3 + i * (w + gap)
        add_box(ax, x, ext_y, w, ext_h, title, sub, fill=BONE, edge=TAUPE)

    fab_top_y = 6.55
    add_box(ax, 0.4, fab_top_y, 3.0, 0.95, "OneLake / Lakehouse",
            "finaq_lake — filings + yfin cache + silver tables", fill=WHITE)
    add_box(ax, 3.55, fab_top_y, 3.0, 0.95, "Warehouse finaq_dw",
            "graph_runs · node_runs · cio_runs · alerts (was state.db)", fill=WHITE)
    add_box(ax, 6.7, fab_top_y, 2.9, 0.95, "Fabric SQL finaq_sql + VECTOR",
            "A/B comparison vs AI Search (phase 9e)", fill=WHITE)
    add_box(ax, 9.75, fab_top_y, 2.9, 0.95, "Eventstream + KQL finaq_kql",
            "Real-time SEC RSS firehose (phase 9c)", fill=WHITE)
    add_box(ax, 12.8, fab_top_y, 2.85, 0.95, "Data Activator",
            "act_critical_risk → /alert (phase 9b)", fill=WHITE)

    fab_mid_y = 5.45
    add_box(ax, 0.4, fab_mid_y, 3.0, 0.95, "Pipeline pl_cio_heartbeat",
            "Data Factory cron 5am+1pm PT → nb_cio", fill=WHITE)
    add_box(ax, 3.55, fab_mid_y, 3.0, 0.95, "Pipeline pl_ingest_universe",
            "Nightly 03:00 PT → nb_ingest", fill=WHITE)
    add_box(ax, 6.7, fab_mid_y, 2.9, 0.95, "Pipeline pl_backtest",
            "On-demand → nb_backtest", fill=WHITE)
    add_box(ax, 9.75, fab_mid_y, 2.9, 0.95, "Power BI Mission Control",
            "Semantic model + RLS demo (phase 9a)", fill=WHITE)
    add_box(ax, 12.8, fab_mid_y, 2.85, 0.95, "AI Skill finaq_copilot",
            "NL Q&A over Lakehouse (phase 9d)", fill=WHITE)

    fab_bot_y = 4.35
    add_box(ax, 0.4, fab_bot_y, 3.0, 0.95, "Notebook nb_cio",
            "cio.dispatcher.main(mode='auto')", fill=WHITE)
    add_box(ax, 3.55, fab_bot_y, 3.0, 0.95, "Notebook nb_ingest",
            "scripts.ingest_universe.main()", fill=WHITE)
    add_box(ax, 6.7, fab_bot_y, 2.9, 0.95, "Notebook nb_backtest",
            "scripts.backtest.main()", fill=WHITE)
    ax.text(12.2, fab_bot_y + 0.47,
            "Notebooks are thin wrappers — they import existing\n"
            "FINAQ Python verbatim from the ACR image",
            fontsize=8, color=INK, ha="center", va="center", style="italic")

    az_top_y = 2.55
    add_box(ax, 0.4, az_top_y, 3.0, 0.95, "Container App: streamlit",
            "ui/app.py + 8 pages — Entra ID auth, ingress=external", fill=EGGSHELL)
    add_box(ax, 3.55, az_top_y, 3.0, 0.95, "Container App: telegram-bot",
            "long-poll listener + FastAPI /alert · min_replicas=1", fill=EGGSHELL)
    add_box(ax, 6.7, az_top_y, 2.9, 0.95, "AI Search finaq-search",
            "Basic SKU — filings + synthesis_reports indexes", fill=EGGSHELL)
    add_box(ax, 9.75, az_top_y, 2.9, 0.95, "AI Foundry finaq-control-tower",
            "project finaq-prod — Anthropic models / role",
            fill=EGGSHELL, existing=True)
    add_box(ax, 12.8, az_top_y, 2.85, 0.95, "Key Vault finaq-kv",
            "Secrets → ACA env at boot", fill=EGGSHELL)

    az_bot_y = 1.0
    add_box(ax, 0.4, az_bot_y, 3.0, 0.95, "Container Registry finaqacr",
            "finaq:latest (one image, two ACA commands)", fill=EGGSHELL)
    add_box(ax, 3.55, az_bot_y, 3.0, 0.95, "Log Analytics + App Insights",
            "ACA stdout, traces, metrics", fill=EGGSHELL)
    add_box(ax, 6.7, az_bot_y, 2.9, 0.95, "Storage Account",
            "ACA logs · AI Search backups", fill=EGGSHELL)
    add_box(ax, 9.75, az_bot_y, 5.9, 0.95, "Managed identities",
            "cae → {kv reader · acr pull · search contributor · "
            "foundry user · OneLake contributor}",
            fill=EGGSHELL)

    # ---------- key arrows (selective for readability) ----------
    # External ingestion sources → notebooks
    add_arrow(ax, (1.55, 8.12), (1.9, 5.30), curve=0.05)        # EDGAR → nb_ingest
    add_arrow(ax, (4.15, 8.12), (3.6, 5.30), curve=-0.05)       # yfinance → nb_ingest
    add_arrow(ax, (6.75, 8.12), (1.9, 5.30), curve=-0.18)       # Finnhub → nb_cio
    add_arrow(ax, (12.0, 8.12), (5.0, 3.50), curve=-0.20)       # Telegram ↔ bot

    # Pipelines → notebooks
    add_arrow(ax, (1.9, 5.45), (1.9, 5.30), curve=0.0)
    add_arrow(ax, (5.05, 5.45), (5.05, 5.30), curve=0.0)
    add_arrow(ax, (8.15, 5.45), (8.15, 5.30), curve=0.0)

    # nb_ingest → OneLake + AI Search
    add_arrow(ax, (4.55, 5.30), (4.55, 6.55), curve=0.0)        # nb_ingest → Warehouse(?) actually OneLake
    add_arrow(ax, (5.55, 5.30), (8.15, 3.50), curve=0.20)       # nb_ingest → AI Search

    # nb_cio → Warehouse + Foundry
    add_arrow(ax, (3.4, 5.10), (5.05, 6.55), curve=-0.10)       # telemetry to Warehouse
    add_arrow(ax, (3.4, 4.65), (11.2, 3.50), curve=0.25)        # LLM to Foundry

    # Streamlit container → reads from everything
    add_arrow(ax, (3.4, 3.0), (6.7, 3.0), curve=0.0)            # → AI Search
    add_arrow(ax, (2.5, 3.50), (5.05, 6.55), curve=-0.28)       # → Warehouse
    add_arrow(ax, (3.0, 3.50), (11.2, 3.50), curve=-0.20)       # → Foundry

    # Warehouse → Power BI semantic model
    add_arrow(ax, (6.55, 7.0), (11.2, 6.40), curve=-0.10)

    # Data Activator → telegram-bot /alert (showcase critical-risk path)
    add_arrow(ax, (13.5, 6.55), (5.0, 3.50), curve=0.35,
              lw=2.2, label="critical-risk path")

    # Key Vault → containers (secrets)
    add_arrow(ax, (12.8, 3.0), (6.55, 3.0), curve=0.0)

    # ACR → container apps (deploy)
    add_arrow(ax, (2.0, 1.95), (2.0, 2.55), curve=0.0)
    add_arrow(ax, (3.0, 1.95), (5.0, 2.55), curve=0.15)

    # ---------- legend ----------
    ax.text(0.4, 0.45, "Legend:", fontsize=9, fontweight="bold", color=INK)

    ax.add_patch(FancyBboxPatch((1.4, 0.30), 0.45, 0.25,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor=EGGSHELL, edgecolor=SAGE, linewidth=1.2))
    ax.text(2.0, 0.43, "new resource (Bicep-provisioned)",
            fontsize=8, color=INK, va="center")

    ax.add_patch(FancyBboxPatch((5.6, 0.30), 0.45, 0.25,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor=EGGSHELL, edgecolor=SAGE, linewidth=1.2,
                                linestyle=(0, (4, 2))))
    ax.text(6.2, 0.43, "existing resource (referenced only)",
            fontsize=8, color=INK, va="center")

    ax.text(9.8, 0.43, "→  dataflow / dependency",
            fontsize=8, color=INK, va="center")
    ax.text(12.6, 0.43, "▬▬▶ showcase critical-risk path",
            fontsize=8, color=SAGE, va="center", fontweight="bold")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=WHITE)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
