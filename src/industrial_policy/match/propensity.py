"""Propensity score estimation."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from industrial_policy.log import get_logger


def build_propensity_scores(
    treated_panel: pd.DataFrame,
    control_pool: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Estimate propensity scores for treated and controls.

    Args:
        treated_panel: Treated event panel.
        control_pool: Control pool panel.
        config: Loaded configuration.

    Returns:
        DataFrame with propensity scores.
    """
    logger = get_logger()
    baseline = config["panel"]["baseline_quarters"]
    covariates = config["analysis"]["control_vars"]

    treated_panel = treated_panel.copy()
    treated_panel["event_date"] = pd.to_datetime(treated_panel["event_date"], errors="coerce")
    control_pool = control_pool.copy()
    control_pool["period_end_date"] = pd.to_datetime(control_pool["period_end_date"], errors="coerce")

    treated_base = treated_panel[
        (treated_panel["event_time_q"] >= baseline["start"]) &
        (treated_panel["event_time_q"] <= baseline["end"])
    ].copy()
    treated_base["event_year"] = treated_base["event_date"].dt.year
    treated_grouped = treated_base.groupby(["cik", "event_year"]).agg({**{col: "mean" for col in covariates}, "sic": "first"})
    treated_grouped = treated_grouped.reset_index()
    treated_grouped["treated"] = 1

    control_pool = control_pool.copy()
    control_pool["event_year"] = control_pool["period_end_date"].dt.year
    control_grouped = control_pool.groupby(["cik", "event_year"]).agg({**{col: "mean" for col in covariates}, "sic": "first"})
    control_grouped = control_grouped.reset_index()
    control_grouped["treated"] = 0

    combined = pd.concat([treated_grouped, control_grouped], ignore_index=True)
    combined = combined.dropna(subset=covariates)

    if combined.empty:
        logger.warning("No data available for propensity score estimation")
        combined["propensity_score"] = np.nan
        return combined
    if combined["treated"].nunique() < 2:
        logger.warning("Propensity score estimation skipped: only one class present.")
        return combined.iloc[0:0].assign(propensity_score=np.nan)

    model = LogisticRegression(max_iter=1000)
    X = combined[covariates]
    y = combined["treated"]
    model.fit(X, y)
    combined["propensity_score"] = model.predict_proba(X)[:, 1]

    return combined
