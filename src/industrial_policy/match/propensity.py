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
    covariates = list(config["analysis"]["control_vars"])
    trend_covariates = ["log_sales", "log_assets", "gross_margin"]
    trend_covariates = [c for c in trend_covariates if c in treated_panel.columns]

    treated_panel = treated_panel.copy()
    treated_panel["event_date"] = pd.to_datetime(treated_panel["event_date"], errors="coerce")
    control_pool = control_pool.copy()
    control_pool["period_end_date"] = pd.to_datetime(control_pool["period_end_date"], errors="coerce")

    treated_base = treated_panel[
        (treated_panel["event_time_q"] >= baseline["start"])
        & (treated_panel["event_time_q"] <= baseline["end"])
    ].copy()
    treated_base["event_year"] = treated_base["event_date"].dt.year

    def _trend_stats(df: pd.DataFrame, value_col: str) -> pd.Series:
        series = pd.to_numeric(df[value_col], errors="coerce")
        mean = series.mean()
        x = pd.to_numeric(df["event_time_q"], errors="coerce").to_numpy(dtype=float)
        y = series.to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return pd.Series({"mean": mean, "slope": 0.0})
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return pd.Series({"mean": mean, "slope": slope})

    def _control_trend_stats(df: pd.DataFrame, value_col: str) -> pd.Series:
        df = df.sort_values("period_end_date")
        series = pd.to_numeric(df[value_col], errors="coerce")
        mean = series.mean()
        y = series.to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return pd.Series({"mean": mean, "slope": 0.0})
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return pd.Series({"mean": mean, "slope": slope})

    treated_stats = {}
    for cov in covariates:
        treated_stats[cov] = "mean"
    treated_grouped = treated_base.groupby(["cik", "event_year"]).agg(
        {**treated_stats, "sic": "first"}
    )
    for cov in trend_covariates:
        trend = treated_base.groupby(["cik", "event_year"]).apply(_trend_stats, value_col=cov)
        trend = trend.reindex(treated_grouped.index)
        treated_grouped[f"{cov}_baseline_mean"] = trend["mean"]
        treated_grouped[f"{cov}_baseline_slope"] = trend["slope"]
        covariates.extend([f"{cov}_baseline_mean", f"{cov}_baseline_slope"])
    treated_grouped = treated_grouped.reset_index()
    treated_grouped["treated"] = 1
    covariates = list(dict.fromkeys(covariates))

    control_pool["event_year"] = control_pool["period_end_date"].dt.year
    if "first_award_date" in control_pool.columns:
        control_pool["first_award_year"] = pd.to_datetime(
            control_pool["first_award_date"], errors="coerce"
        ).dt.year
        control_pool = control_pool[
            control_pool["first_award_year"].isna()
            | (control_pool["first_award_year"] > control_pool["event_year"])
        ]
    control_grouped = control_pool.groupby(["cik", "event_year"]).agg(
        {**{col: "mean" for col in covariates if col in control_pool.columns}, "sic": "first"}
    )
    for cov in trend_covariates:
        if cov not in control_pool.columns:
            continue
        trend = control_pool.groupby(["cik", "event_year"]).apply(
            _control_trend_stats, value_col=cov
        )
        trend = trend.reindex(control_grouped.index)
        control_grouped[f"{cov}_baseline_mean"] = trend["mean"]
        control_grouped[f"{cov}_baseline_slope"] = trend["slope"]
    control_grouped = control_grouped.reset_index()
    control_grouped["treated"] = 0

    combined = pd.concat([treated_grouped, control_grouped], ignore_index=True)
    combined = combined.dropna(subset=[col for col in covariates if col in combined.columns])

    if combined.empty:
        logger.warning("No data available for propensity score estimation")
        combined["propensity_score"] = np.nan
        return combined
    if combined["treated"].nunique() < 2:
        logger.warning("Propensity score estimation skipped: only one class present.")
        return combined.iloc[0:0].assign(propensity_score=np.nan)

    model = LogisticRegression(max_iter=1000)
    X = combined[[col for col in covariates if col in combined.columns]]
    y = combined["treated"]
    model.fit(X, y)
    combined["propensity_score"] = model.predict_proba(X)[:, 1]

    return combined
