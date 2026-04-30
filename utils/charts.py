"""Palette-aware chart rendering for the dashboard + PDF report.

All charts in the FINAQ dashboard and PDF report use the same warm-neutral
editorial palette (CLAUDE.md §13). Living in one module so:
  - The Streamlit UI and PDF exporter render identical visuals.
  - Color drift between surfaces is impossible (one source of truth).
  - Future palette tweaks are one-edit-only.

Public:
  - `mc_histogram(samples, current_price, percentiles=...)` returns a
    matplotlib Figure ready for Streamlit `st.pyplot(...)` or PDF embed.
  - `mc_histogram_to_png(samples, ..., output_path)` writes a PNG file
    suitable for ReportLab `Image(...)` flowables.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import matplotlib

# Force a non-interactive backend so the module is safe to import in headless
# contexts (PDF export from a worker, server-side rendering). Streamlit
# overrides this when needed; PDF export benefits from it.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# --- Palette (CLAUDE.md §13) ------------------------------------------------

SAGE = "#2D4F3A"
PARCHMENT = "#F4ECDC"
WHITE = "#FFFFFF"
EGGSHELL = "#FBF5E8"
TAUPE = "#E0D5C2"
BONE = "#EDE5D5"
INK = "#1A1611"


# --- Monte Carlo histogram --------------------------------------------------


_DEFAULT_PERCENTILES = (10, 25, 50, 75, 90)


def mc_histogram(
    samples: Sequence[float],
    current_price: float | None = None,
    title: str = "Monte Carlo fair-value distribution",
    bins: int = 60,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> plt.Figure:
    """Render the MC sample distribution with parchment fill + sage edge,
    P10/P50/P90 percentile lines (taupe), and a dashed current-price line (ink).

    The chart deliberately omits gridlines and decorative furniture — the
    palette + percentile annotations carry the meaning.
    """
    arr = np.asarray(list(samples), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        # Empty distribution — render an explicit "no data" placeholder rather
        # than crash. The PDF and Streamlit will both surface this cleanly.
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(WHITE)
        ax.text(0.5, 0.5, "No Monte Carlo samples available", color=INK,
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return fig

    p10, p25, p50, p75, p90 = np.percentile(arr, _DEFAULT_PERCENTILES)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    ax.hist(
        arr,
        bins=bins,
        color=PARCHMENT,
        edgecolor=SAGE,
        linewidth=0.7,
    )

    # P10 / P90 in taupe
    for value in (p10, p90):
        ax.axvline(value, color=TAUPE, linewidth=1.4, linestyle="-")
    # P50 (median) in sage — the load-bearing percentile
    ax.axvline(p50, color=SAGE, linewidth=2.0, linestyle="-")
    # Current price in ink, dashed — the "where we are" marker
    if current_price is not None and np.isfinite(current_price):
        ax.axvline(float(current_price), color=INK, linewidth=1.6, linestyle="--")

    # Labels above the percentile lines
    y_top = ax.get_ylim()[1]
    label_y = y_top * 0.95
    for label, value, color in (
        ("P10", p10, TAUPE),
        ("P50", p50, SAGE),
        ("P90", p90, TAUPE),
    ):
        ax.text(value, label_y, f"  {label}\n  ${value:,.0f}", color=color,
                fontsize=8, va="top", ha="left", fontweight="bold")
    if current_price is not None and np.isfinite(current_price):
        ax.text(float(current_price), label_y * 0.7,
                f"  current\n  ${current_price:,.0f}", color=INK,
                fontsize=8, va="top", ha="left", fontweight="bold")

    # Strip distracting spines/ticks
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BONE)
    ax.spines["left"].set_color(BONE)
    ax.tick_params(axis="x", colors=INK, labelsize=9)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    ax.set_xlabel("Fair value per share ($)", color=INK, fontsize=10)
    ax.set_ylabel("Simulations", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)

    fig.tight_layout()
    return fig


def mc_histogram_to_png(
    samples: Sequence[float],
    output_path: str | Path,
    current_price: float | None = None,
    title: str = "Monte Carlo fair-value distribution",
    dpi: int = 150,
) -> Path:
    """Render the MC histogram and save to PNG. Used by the PDF exporter."""
    fig = mc_histogram(samples, current_price=current_price, title=title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return output_path


def mc_histogram_to_bytes(
    samples: Sequence[float],
    current_price: float | None = None,
    title: str = "Monte Carlo fair-value distribution",
    dpi: int = 150,
) -> bytes:
    """Render the MC histogram into an in-memory PNG byte buffer. Useful for
    Streamlit (`st.image`) and tests where we don't want to touch disk."""
    fig = mc_histogram(samples, current_price=current_price, title=title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return buf.getvalue()
