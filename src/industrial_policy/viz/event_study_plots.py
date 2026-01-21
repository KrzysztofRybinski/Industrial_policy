"""Event study plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from industrial_policy.log import get_logger


def plot_event_study(coef_df: pd.DataFrame, outcome: str, outputs_dir: str) -> Path:
    """Plot event-study coefficients with confidence intervals.

    Args:
        coef_df: Coefficient dataframe.
        outcome: Outcome name.
        outputs_dir: Base outputs directory.

    Returns:
        Path to saved plot.
    """
    logger = get_logger()
    figures_dir = Path(outputs_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f"event_study_{outcome}.png"

    plt.figure(figsize=(8, 4))
    plt.errorbar(
        coef_df["event_time_q"],
        coef_df["coef"],
        yerr=1.96 * coef_df["se"],
        fmt="o",
        color="tab:blue",
        ecolor="lightgray",
    )
    plt.axvline(-1, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle=":", linewidth=1)
    plt.title(f"Event Study: {outcome}")
    plt.xlabel("Event time (quarters)")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    logger.info("Saved plot %s", fig_path)
    return fig_path
