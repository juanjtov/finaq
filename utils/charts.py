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


# --- Backtest time-series chart --------------------------------------------


def backtest_price_path(
    as_of_date: str,
    as_of_close: float | None,
    realised: list[tuple[int, str, float | None]],
    mc_percentiles: dict[str, float] | None = None,
    *,
    title: str = "Realised price vs Monte Carlo fair-value bands",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> plt.Figure:
    """Render a backtest time-series chart: as-of price + realised closes at
    each horizon, overlaid against horizontal bands of the MC fair-value
    distribution.

    Args:
      as_of_date:    ISO date string the drill-in was anchored to.
      as_of_close:   price the day of the drill-in (= mc.current_price).
      realised:      list of (horizon_days, target_date_iso, close) tuples
                     for each scoring horizon (30 / 90 / 180 d).
      mc_percentiles: dict like {"p10": .., "p25": .., "p50": .., "p75":
                     .., "p90": ..} from the DCF distribution. Drawn as
                     horizontal reference bands. Optional.

    Visualises:
      - Sage P50 horizontal line.
      - Parchment-filled P10-P90 band, eggshell-filled P25-P75 band.
      - Ink dashed line for the as-of close.
      - Markers (sage circles) for each realised horizon close, with text
        labels showing %-return from as-of.
    """
    from datetime import datetime as _dt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    # Convert dates to datetimes for matplotlib's date-axis.
    try:
        x_as_of = _dt.fromisoformat(as_of_date)
    except ValueError:
        x_as_of = None

    realised_points: list[tuple[_dt, float, int]] = []
    for days, iso, close in realised:
        if close is None:
            continue
        try:
            xt = _dt.fromisoformat(iso)
        except ValueError:
            continue
        realised_points.append((xt, float(close), int(days)))

    # Horizontal bands first (so they sit behind the price markers).
    # Sage is the brand accent — using two opacities of sage as a single-hue
    # gradient gives strong visual distinction between the outer P10-P90
    # band and the inner P25-P75 band while staying inside the warm-neutral
    # palette. Earlier attempt with parchment/eggshell was too low-contrast
    # to read at small chart sizes.
    if mc_percentiles:
        p10 = mc_percentiles.get("p10")
        p25 = mc_percentiles.get("p25")
        p50 = mc_percentiles.get("p50")
        p75 = mc_percentiles.get("p75")
        p90 = mc_percentiles.get("p90")
        # X-extent: as-of through last realised horizon (or +30 fallback)
        x_left = x_as_of
        x_right = realised_points[-1][0] if realised_points else x_as_of
        if x_left is not None and x_right is not None and x_right >= x_left:
            if p10 is not None and p90 is not None:
                ax.fill_between(
                    [x_left, x_right], [p10, p10], [p90, p90],
                    color=SAGE, alpha=0.10, label="MC P10-P90",
                )
            if p25 is not None and p75 is not None:
                ax.fill_between(
                    [x_left, x_right], [p25, p25], [p75, p75],
                    color=SAGE, alpha=0.30, label="MC P25-P75",
                )
            if p50 is not None:
                ax.hlines(
                    p50, x_left, x_right, colors=SAGE, linestyles="-",
                    linewidth=1.8, label=f"MC P50 = ${p50:,.0f}",
                )

    # As-of price line (ink dashed)
    if x_as_of is not None and as_of_close is not None:
        ax.axhline(
            float(as_of_close), color=INK, linestyle="--", linewidth=1.2,
            alpha=0.55,
        )
        ax.scatter(
            [x_as_of], [float(as_of_close)], color=INK, s=55, zorder=5,
            label=f"as-of {as_of_date} = ${as_of_close:,.2f}",
        )

    # Realised horizon closes (sage circles + return labels)
    if realised_points:
        xs = [p[0] for p in realised_points]
        ys = [p[1] for p in realised_points]
        ax.plot(xs, ys, color=SAGE, marker="o", linewidth=1.4, zorder=4)
        for xt, close, days in realised_points:
            if as_of_close:
                pct = (close - float(as_of_close)) / float(as_of_close) * 100
                lbl = f"+{days}d  ${close:,.2f}\n{pct:+.1f}%"
            else:
                lbl = f"+{days}d  ${close:,.2f}"
            ax.annotate(
                lbl, xy=(xt, close),
                xytext=(6, 8), textcoords="offset points",
                color=INK, fontsize=8, fontweight="bold",
            )

    # Strip distracting spines/ticks
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BONE)
    ax.spines["left"].set_color(BONE)
    ax.tick_params(axis="x", colors=INK, labelsize=9)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    ax.set_xlabel("Date", color=INK, fontsize=10)
    ax.set_ylabel("Price ($)", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    if mc_percentiles or realised_points:
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


# --- Sample resolution (shared between dashboard + Telegram bot) ----------


_MC_HISTOGRAM_SEED = 42  # mirrors utils.monte_carlo.simulate's default seed


def resolve_mc_samples(mc: dict) -> list[float]:
    """Return the MC sample array for histogram rendering.

    Saved demo files store percentiles + metadata but not the raw 10k-sample
    array (size optimisation). When a caller needs to redraw the histogram —
    Streamlit dashboard rerun, Telegram bot photo upload, future PDF re-export
    — we regenerate a visually-similar normal distribution from the
    P10/P50/P90 spread using a fixed RNG seed. Same cached state → same
    chart bars across surfaces, no jitter.

    Returns an empty list when neither raw samples nor DCF percentiles are
    present (e.g. MC was skipped because shares_outstanding was missing —
    see ARCHITECTURE §6.10).
    """
    samples = mc.get("samples")
    if samples is not None and len(samples) > 0:
        return list(samples)
    dcf = mc.get("dcf") or {}
    if dcf:
        lo = dcf.get("p10") or 0
        hi = dcf.get("p90") or 0
        mid = dcf.get("p50") or (lo + hi) / 2
        # Std derived from P10–P90 spread assuming roughly normal: 2.563σ
        # covers the 80% interval, so spread / 2.6 ≈ σ.
        std = max((hi - lo) / 2.6, 1.0)
        rng = np.random.default_rng(_MC_HISTOGRAM_SEED)
        return list(rng.normal(loc=mid, scale=std, size=8000))
    return []
