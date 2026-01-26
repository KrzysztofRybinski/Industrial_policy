"""Incidence-in-dollars analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from industrial_policy.log import get_logger


def _load_event_betas(outputs_dir: Path, outcome: str) -> dict[int, float]:
    path = outputs_dir / "tables" / f"event_study_{outcome}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row["event_time_q"]): float(row["beta"]) for _, row in df.iterrows()}


def run_incidence_analysis(config: Dict[str, Any]) -> None:
    """Compute incidence ratios for treated firms."""
    logger = get_logger()
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    outputs_dir = Path(config["project"]["outputs_dir"])
    stacked_path = data_dir / "event_panel_stacked.parquet"
    if not stacked_path.exists():
        logger.warning("Stacked panel not found; run build panel first.")
        return

    stacked = pd.read_parquet(stacked_path)
    if stacked.empty:
        logger.warning("Stacked panel empty; skipping incidence.")
        return

    gm_betas = _load_event_betas(outputs_dir, "gross_margin")
    capex_betas = _load_event_betas(outputs_dir, "capex_intensity")
    if not gm_betas or not capex_betas:
        logger.warning("Missing event-study betas; run estimate first.")
        return

    treated = stacked[stacked["treated_in_stack"] == 1].copy()
    treated["award_amount"] = pd.to_numeric(treated.get("first_award_amount"), errors="coerce")
    treated["revenue"] = pd.to_numeric(treated.get("revenue"), errors="coerce")
    treated = treated.dropna(subset=["award_amount", "revenue", "event_time_q"])

    if treated.empty:
        logger.warning("No treated firm rows with revenue and awards.")
        return

    horizons = [2, 6, 12]
    records = []
    for stack_id, group in treated.groupby("stack_id"):
        award_amount = group["award_amount"].iloc[0]
        if award_amount <= 0:
            continue
        for horizon in horizons:
            subset = group[(group["event_time_q"] >= 0) & (group["event_time_q"] <= horizon)]
            if subset.empty:
                continue
            gm_increment = 0.0
            capex_increment = 0.0
            for _, row in subset.iterrows():
                beta_gm = gm_betas.get(int(row["event_time_q"]), 0.0)
                beta_capex = capex_betas.get(int(row["event_time_q"]), 0.0)
                gm_increment += beta_gm * row["revenue"]
                capex_increment += beta_capex * row["revenue"]
            records.append(
                {
                    "stack_id": stack_id,
                    "horizon": horizon,
                    "incremental_gross_profit": gm_increment,
                    "incremental_capex": capex_increment,
                    "award_amount": award_amount,
                    "gross_profit_per_award": gm_increment / award_amount,
                    "capex_per_award": capex_increment / award_amount,
                }
            )

    firm_df = pd.DataFrame(records)
    if firm_df.empty:
        logger.warning("No incidence ratios computed.")
        return

    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    firm_path = tables_dir / "incidence_ratios_firmlevel.csv"
    firm_df.to_csv(firm_path, index=False)

    summary = (
        firm_df.groupby("horizon")[["gross_profit_per_award", "capex_per_award"]]
        .agg(["mean", "median"])
        .reset_index()
    )
    summary.columns = ["horizon", "gross_profit_mean", "gross_profit_median", "capex_mean", "capex_median"]
    summary_path = tables_dir / "incidence_ratios_summary.csv"
    summary.to_csv(summary_path, index=False)
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(firm_df["gross_profit_per_award"].dropna(), bins=30, alpha=0.7)
    plt.xlabel("Incremental gross profit / award")
    plt.ylabel("Count")
    plt.title("Incidence ratios (gross profit per award)")
    plt.tight_layout()
    plt.savefig(figures_dir / "incidence_gross_profit_hist.png", dpi=150)
    plt.close()
    logger.info("Saved incidence outputs to %s", tables_dir)
