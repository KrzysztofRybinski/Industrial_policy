"""Matching diagnostics outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from industrial_policy.log import get_logger


def _weighted_mean_var(values: np.ndarray, weights: Optional[np.ndarray]) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if weights is None:
        mean = float(np.mean(values))
        var = float(np.var(values, ddof=0))
    else:
        mean = float(np.average(values, weights=weights))
        var = float(np.average((values - mean) ** 2, weights=weights))
    return mean, var


def _smd(
    treated: np.ndarray,
    control: np.ndarray,
    treated_weights: Optional[np.ndarray] = None,
    control_weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    t_mean, t_var = _weighted_mean_var(treated, treated_weights)
    c_mean, c_var = _weighted_mean_var(control, control_weights)
    pooled_sd = np.sqrt((t_var + c_var) / 2) if not np.isnan(t_var + c_var) else float("nan")
    if pooled_sd == 0 or np.isnan(pooled_sd):
        smd = float("nan")
    else:
        smd = (t_mean - c_mean) / pooled_sd
    return t_mean, c_mean, smd


def _propensity_paths(config: Dict[str, Any]) -> Tuple[Path, Path]:
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    return data_dir / "propensity_scores.parquet", data_dir / "matches.parquet"


def write_matching_diagnostics(config: Dict[str, Any], outputs_dir: str | Path) -> None:
    """Write matching diagnostics tables and plots if data are available."""
    logger = get_logger()
    propensity_path, matches_path = _propensity_paths(config)
    if not propensity_path.exists():
        logger.info("Propensity scores not found; skipping matching diagnostics")
        return

    propensity = pd.read_parquet(propensity_path)
    if propensity.empty:
        logger.info("Propensity scores empty; skipping matching diagnostics")
        return

    covariates: Iterable[str] = config["analysis"]["control_vars"]
    tables_dir = Path(outputs_dir) / "tables"
    figures_dir = Path(outputs_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    before_treated = propensity[propensity["treated"] == 1]
    before_control = propensity[propensity["treated"] == 0]

    after_treated = pd.DataFrame()
    after_control = pd.DataFrame()
    control_weights = None
    if matches_path.exists():
        matches = pd.read_parquet(matches_path)
        treated_keys = matches[["treated_cik", "treated_year"]].drop_duplicates()
        after_treated = treated_keys.merge(
            propensity,
            left_on=["treated_cik", "treated_year"],
            right_on=["cik", "event_year"],
            how="left",
        )
        control_keys = matches[["control_cik", "control_year", "weight"]]
        after_control = control_keys.merge(
            propensity,
            left_on=["control_cik", "control_year"],
            right_on=["cik", "event_year"],
            how="left",
        )
        control_weights = after_control["weight"].to_numpy()

    rows = []
    for covariate in covariates:
        if covariate not in propensity.columns:
            continue
        t_vals = before_treated[covariate].dropna().to_numpy()
        c_vals = before_control[covariate].dropna().to_numpy()
        t_mean, c_mean, smd_before = _smd(t_vals, c_vals)

        t_after_vals = (
            after_treated[covariate].dropna().to_numpy()
            if not after_treated.empty
            else np.array([])
        )
        c_after_vals = (
            after_control[covariate].dropna().to_numpy()
            if not after_control.empty
            else np.array([])
        )
        c_weights = None
        if control_weights is not None and c_after_vals.size:
            c_weights = after_control.loc[
                after_control[covariate].notna(), "weight"
            ].to_numpy()
        t_after_mean, c_after_mean, smd_after = _smd(
            t_after_vals,
            c_after_vals,
            None,
            c_weights,
        )

        rows.append(
            {
                "covariate": covariate,
                "mean_treated_before": t_mean,
                "mean_control_before": c_mean,
                "smd_before": smd_before,
                "mean_treated_after": t_after_mean,
                "mean_control_after": c_after_mean,
                "smd_after": smd_after,
            }
        )

    balance_df = pd.DataFrame(rows)
    balance_path = tables_dir / "match_balance.csv"
    balance_df.to_csv(balance_path, index=False)
    logger.info("Saved match balance table to %s", balance_path)

    if "propensity_score" in propensity.columns:
        treated_scores = propensity.loc[propensity["treated"] == 1, "propensity_score"].dropna()
        control_scores = propensity.loc[propensity["treated"] == 0, "propensity_score"].dropna()
        if not treated_scores.empty and not control_scores.empty:
            plt.figure(figsize=(6, 4))
            plt.hist(treated_scores, bins=30, alpha=0.6, label="Treated")
            plt.hist(control_scores, bins=30, alpha=0.6, label="Control")
            plt.xlabel("Propensity score")
            plt.ylabel("Count")
            plt.title("Propensity Score Overlap")
            plt.legend()
            overlap_path = figures_dir / "pscore_overlap.png"
            plt.tight_layout()
            plt.savefig(overlap_path, dpi=150)
            plt.close()
            logger.info("Saved propensity overlap plot to %s", overlap_path)
