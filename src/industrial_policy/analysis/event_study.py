"""Event study estimation."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from industrial_policy.log import get_logger


def build_stacked_dataset(
    treated_panel: pd.DataFrame,
    control_pool: pd.DataFrame,
    matches: pd.DataFrame,
    window: Dict[str, int],
) -> pd.DataFrame:
    """Build stacked dataset with matched controls aligned to event time.

    Args:
        treated_panel: Treated event panel.
        control_pool: Control pool panel.
        matches: Matching table.
        window: Event window configuration.

    Returns:
        Stacked event-study dataset.
    """
    logger = get_logger()
    stacks = []

    for treated_cik in matches["treated_cik"].unique():
        treated_firm = treated_panel[treated_panel["cik"] == treated_cik].copy()
        if treated_firm.empty:
            continue
        event_date = treated_firm["event_date"].iloc[0]
        treated_firm["stack_id"] = treated_cik
        treated_firm["weight"] = 1.0
        stacks.append(treated_firm)

        matched_controls = matches[matches["treated_cik"] == treated_cik]
        for _, match in matched_controls.iterrows():
            control_firm = control_pool[control_pool["cik"] == match["control_cik"]].copy()
            if control_firm.empty:
                continue
            control_firm = control_firm.sort_values("period_end_date").reset_index(drop=True)
            event_period = control_firm[control_firm["period_end_date"] >= event_date]
            if event_period.empty:
                continue
            event_index = event_period.index[0]
            control_firm["event_time_q"] = control_firm.index - event_index
            control_firm = control_firm[
                (control_firm["event_time_q"] >= -window["pre"]) &
                (control_firm["event_time_q"] <= window["post"])
            ]
            control_firm["treated"] = 0
            control_firm["event_date"] = event_date
            control_firm["stack_id"] = treated_cik
            control_firm["weight"] = match["weight"]
            stacks.append(control_firm)

    if not stacks:
        logger.warning("No stacks created for event study")
        return pd.DataFrame()
    stacked = pd.concat(stacks, ignore_index=True)
    return stacked


def estimate_event_study(
    stacked: pd.DataFrame,
    outcome: str,
    control_vars: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Estimate event study coefficients with firm and time fixed effects.

    Args:
        stacked: Stacked event-study data.
        outcome: Outcome variable name.
        control_vars: Additional controls.

    Returns:
        Tuple of (coefficients dataframe, metadata).
    """
    logger = get_logger()
    if outcome not in stacked.columns:
        logger.warning("Outcome %s not found in stacked data", outcome)
        return pd.DataFrame(), {}
    df = stacked.dropna(subset=[outcome]).copy()
    if df.empty:
        return pd.DataFrame(), {}

    df["event_time_q"] = df["event_time_q"].astype(int)
    event_dummies = pd.get_dummies(df["event_time_q"], prefix="event", drop_first=False)
    if "event_-1" in event_dummies.columns:
        event_dummies = event_dummies.drop(columns=["event_-1"])
    event_dummies = event_dummies.mul(df["treated"].values, axis=0)

    firm_dummies = pd.get_dummies(df["cik"], prefix="firm", drop_first=True)
    time_dummies = pd.get_dummies(df["event_time_q"], prefix="time", drop_first=True)

    X = pd.concat([event_dummies, df[control_vars], firm_dummies, time_dummies], axis=1)
    X = sm.add_constant(X)

    model = sm.WLS(df[outcome], X, weights=df.get("weight", 1.0))
    results = model.fit(cov_type="cluster", cov_kwds={"groups": df["cik"]})

    coeffs = []
    for name, value in results.params.items():
        if name.startswith("event_"):
            pvalue = results.pvalues.get(name, float("nan"))
            coeffs.append(
                {
                    "event_time_q": int(name.split("_")[-1]),
                    "beta": value,
                    "se": results.bse[name],
                    "pvalue": pvalue,
                }
            )
    coef_df = pd.DataFrame(coeffs).sort_values("event_time_q")
    coef_df["ci_low"] = coef_df["beta"] - 1.96 * coef_df["se"]
    coef_df["ci_high"] = coef_df["beta"] + 1.96 * coef_df["se"]

    pretrend_names = []
    for col in event_dummies.columns:
        try:
            event_time = int(col.split("_")[-1])
        except ValueError:
            continue
        if event_time <= -2 and col in results.params.index:
            pretrend_names.append(col)
    pretrend_pvalue = None
    if pretrend_names:
        constraints = ", ".join([f"{name} = 0" for name in pretrend_names])
        try:
            pretrend_pvalue = float(results.f_test(constraints).pvalue)
        except Exception:  # pragma: no cover - guard against singular models
            pretrend_pvalue = None

    metadata = {
        "n_obs": float(len(df)),
        "n_firms": float(df["cik"].nunique()),
        "pretrend_pvalue": pretrend_pvalue,
    }
    return coef_df, metadata
