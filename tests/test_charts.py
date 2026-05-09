"""Tier 1 unit tests for utils/charts.py — palette-aware MC histogram.

These tests verify the chart helpers don't crash on empty / malformed
samples and produce non-empty PNG output. Visual fidelity is not checked
deterministically (palette correctness is enforced by the constants in
the module — if the constants drift, the chart drifts too).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.charts import mc_histogram, mc_histogram_to_bytes, mc_histogram_to_png


def test_mc_histogram_returns_a_figure_for_normal_input():
    samples = np.random.normal(loc=185, scale=40, size=1000)
    fig = mc_histogram(samples, current_price=200.0)
    assert fig is not None
    # Must close to free memory in CI loops
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_mc_histogram_handles_empty_samples():
    """Empty array → placeholder figure, NOT a crash."""
    fig = mc_histogram([], current_price=200.0)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_mc_histogram_handles_all_nan_samples():
    """All-NaN → empty distribution → placeholder."""
    fig = mc_histogram([float("nan"), float("nan")], current_price=200.0)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_mc_histogram_handles_missing_current_price():
    """`current_price=None` should skip the dashed line, not raise."""
    samples = np.random.normal(loc=185, scale=40, size=500)
    fig = mc_histogram(samples, current_price=None)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_mc_histogram_to_bytes_produces_png_signature():
    """The PNG byte buffer must start with the 8-byte PNG magic."""
    samples = np.random.normal(loc=185, scale=40, size=500)
    data = mc_histogram_to_bytes(samples, current_price=200.0)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # And actually have some payload (not just header)
    assert len(data) > 1000


def test_mc_histogram_to_png_writes_file_and_returns_path(tmp_path: Path):
    samples = np.random.normal(loc=185, scale=40, size=500)
    out = mc_histogram_to_png(samples, tmp_path / "h.png", current_price=200.0)
    assert out == tmp_path / "h.png"
    assert out.exists()
    assert out.stat().st_size > 1000


def test_mc_histogram_to_png_creates_parent_dirs(tmp_path: Path):
    samples = np.random.normal(loc=185, scale=40, size=500)
    nested = tmp_path / "a" / "b" / "h.png"
    out = mc_histogram_to_png(samples, nested, current_price=200.0)
    assert out.exists()


@pytest.mark.parametrize("size", [10, 100, 10_000])
def test_mc_histogram_handles_various_sample_counts(size):
    samples = np.random.normal(loc=185, scale=40, size=size)
    data = mc_histogram_to_bytes(samples, current_price=200.0)
    assert len(data) > 500


# --- backtest_price_path ---------------------------------------------------


def test_backtest_price_path_returns_figure_for_full_input():
    from utils.charts import backtest_price_path

    fig = backtest_price_path(
        as_of_date="2025-09-05",
        as_of_close=21.85,
        realised=[
            (30, "2025-10-05", 22.10),
            (90, "2025-12-04", 19.80),
            (180, "2026-03-04", 24.90),
        ],
        mc_percentiles={"p10": 12, "p25": 16, "p50": 24, "p75": 32, "p90": 45},
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_backtest_price_path_handles_partial_data():
    """Missing realised closes (yfinance gaps) → still renders, no crash."""
    from utils.charts import backtest_price_path

    fig = backtest_price_path(
        as_of_date="2025-09-05",
        as_of_close=None,
        realised=[(30, "2025-10-05", None)],  # nothing realised
        mc_percentiles=None,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_backtest_price_path_handles_no_mc_percentiles():
    """No MC bands → just the price markers, no fill_between."""
    from utils.charts import backtest_price_path

    fig = backtest_price_path(
        as_of_date="2025-09-05",
        as_of_close=21.85,
        realised=[(30, "2025-10-05", 22.10)],
        mc_percentiles=None,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
