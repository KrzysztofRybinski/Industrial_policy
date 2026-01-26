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
        treated_firm["calendar_quarter"] = pd.to_datetime(
            treated_firm["period_end_date"], errors="coerce"
        ).dt.to_period("Q")
        treated_firm["stack_id"] = treated_cik
        treated_firm["weight"] = 1.0
        treated_firm["treated_in_stack"] = 1
        stacks.append(treated_firm)

        matched_controls = matches[matches["treated_cik"] == treated_cik]
        for _, match in matched_controls.iterrows():
            control_firm = control_pool[control_pool["cik"] == match["control_cik"]].copy()
            if control_firm.empty:
                continue
            if "first_award_date" in control_firm.columns:
                first_award = pd.to_datetime(control_firm["first_award_date"].iloc[0], errors="coerce")
                if pd.notna(first_award) and first_award <= event_date:
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
            if "first_award_date" in control_firm.columns:
                control_firm = control_firm[
                    control_firm["period_end_date"] < pd.to_datetime(
                        control_firm["first_award_date"], errors="coerce"
                    )
                ]
            control_firm["calendar_quarter"] = pd.to_datetime(
                control_firm["period_end_date"], errors="coerce"
            ).dt.to_period("Q")
            control_firm["treated"] = 0
            control_firm["event_date"] = event_date
            control_firm["stack_id"] = treated_cik
            control_firm["weight"] = match["weight"]
            control_firm["treated_in_stack"] = 0
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
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")

    df["event_time_q"] = df["event_time_q"].astype(int)
    control_vars = [col for col in control_vars if col in df.columns]
    if "calendar_quarter" not in df.columns and "period_end_date" in df.columns:
        df["calendar_quarter"] = pd.to_datetime(
            df["period_end_date"], errors="coerce"
        ).dt.to_period("Q")
    event_dummies = pd.get_dummies(df["event_time_q"], prefix="event", drop_first=False)
    if "event_-1" in event_dummies.columns:
        event_dummies = event_dummies.drop(columns=["event_-1"])
    event_dummies = event_dummies.mul(df["treated_in_stack"].values, axis=0)

    df["stack_unit_id"] = df["stack_id"].astype(str) + ":" + df["cik"].astype(str)
    df["stack_time_id"] = df["stack_id"].astype(str) + ":" + df["calendar_quarter"].astype(str)

    unit_fe = pd.get_dummies(df["stack_unit_id"], prefix="unit", drop_first=True)
    time_fe = pd.get_dummies(df["stack_time_id"], prefix="time", drop_first=True)

    control_frame = df[control_vars] if control_vars else pd.DataFrame(index=df.index)
    X = pd.concat([event_dummies, control_frame, unit_fe, time_fe], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = X.apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1) & df[outcome].notna()
    df = df.loc[valid_rows].copy()
    X = X.loc[valid_rows].astype(float)
    weights = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(1.0)

    variance = X.var(axis=0)
    zero_cols = [col for col in variance[variance == 0].index.tolist() if col != "const"]
    if zero_cols:
        X = X.drop(columns=zero_cols)

    try:
        model = sm.WLS(df[outcome], X, weights=weights)
        results = model.fit(cov_type="cluster", cov_kwds={"groups": df["cik"]})
    except Exception as exc:  # pragma: no cover
        logger.warning("Event study fit singular for %s: %s", outcome, exc)
        results = None

    coeffs = []
    if results is not None:
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
    else:
        for name in event_dummies.columns:
            if name.startswith("event_"):
                coeffs.append(
                    {
                        "event_time_q": int(name.split("_")[-1]),
                        "beta": float("nan"),
                        "se": float("nan"),
                        "pvalue": float("nan"),
                    }
                )
    coef_df = pd.DataFrame(coeffs).sort_values("event_time_q")
    coef_df["ci_low"] = coef_df["beta"] - 1.96 * coef_df["se"]
    coef_df["ci_high"] = coef_df["beta"] + 1.96 * coef_df["se"]

    pretrend_pvalue = None
    if results is not None:
        pretrend_names = []
        for col in event_dummies.columns:
            try:
                event_time = int(col.split("_")[-1])
            except ValueError:
                continue
            if event_time <= -2 and col in results.params.index:
                pretrend_names.append(col)
        if pretrend_names:
            constraints = ", ".join([f"{name} = 0" for name in pretrend_names])
            try:
                pretrend_pvalue = float(results.f_test(constraints).pvalue)
            except Exception:  # pragma: no cover
                pretrend_pvalue = None

    metadata = {
        "n_obs": float(len(df)),
        "n_firms": float(df["cik"].nunique()),
        "pretrend_pvalue": pretrend_pvalue,
    }
    return coef_df, metadata


def estimate_event_study_interaction(
    stacked: pd.DataFrame,
    outcome: str,
    control_vars: List[str],
    interaction_col: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Estimate event study with treated and treated*interaction terms."""
    logger = get_logger()
    if interaction_col not in stacked.columns:
        logger.warning("Interaction column %s not found; skipping.", interaction_col)
        return pd.DataFrame(), {}
    df = stacked.dropna(subset=[outcome]).copy()
    if df.empty:
        return pd.DataFrame(), {}
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")

    df["event_time_q"] = df["event_time_q"].astype(int)
    control_vars = [col for col in control_vars if col in df.columns]
    if "calendar_quarter" not in df.columns and "period_end_date" in df.columns:
        df["calendar_quarter"] = pd.to_datetime(
            df["period_end_date"], errors="coerce"
        ).dt.to_period("Q")
    base_dummies = pd.get_dummies(df["event_time_q"], prefix="event", drop_first=False)
    if "event_-1" in base_dummies.columns:
        base_dummies = base_dummies.drop(columns=["event_-1"])
    treated_terms = base_dummies.mul(df["treated_in_stack"].values, axis=0)
    interaction_terms = base_dummies.mul(
        df["treated_in_stack"].values * df[interaction_col].fillna(0).values, axis=0
    )
    interaction_terms = interaction_terms.add_suffix("_x_hhi")

    df["stack_unit_id"] = df["stack_id"].astype(str) + ":" + df["cik"].astype(str)
    df["stack_time_id"] = df["stack_id"].astype(str) + ":" + df["calendar_quarter"].astype(str)

    unit_fe = pd.get_dummies(df["stack_unit_id"], prefix="unit", drop_first=True)
    time_fe = pd.get_dummies(df["stack_time_id"], prefix="time", drop_first=True)
    control_frame = df[control_vars] if control_vars else pd.DataFrame(index=df.index)

    X = pd.concat([treated_terms, interaction_terms, control_frame, unit_fe, time_fe], axis=1)
    X = sm.add_constant(X, has_constant="add").apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1) & df[outcome].notna()
    df = df.loc[valid_rows].copy()
    X = X.loc[valid_rows].astype(float)
    weights = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(1.0)

    variance = X.var(axis=0)
    zero_cols = [col for col in variance[variance == 0].index.tolist() if col != "const"]
    if zero_cols:
        X = X.drop(columns=zero_cols)

    try:
        model = sm.WLS(df[outcome], X, weights=weights)
        results = model.fit(cov_type="cluster", cov_kwds={"groups": df["cik"]})
    except Exception as exc:  # pragma: no cover
        logger.warning("Interaction model singular for %s: %s", outcome, exc)
        results = None

    coeffs = []
    if results is not None:
        for name, value in results.params.items():
            if name.startswith("event_") and not name.endswith("_x_hhi"):
                term = "treated"
                event_str = name.replace("event_", "")
            elif name.startswith("event_") and name.endswith("_x_hhi"):
                term = "treated_x_hhi"
                event_str = name.replace("event_", "").replace("_x_hhi", "")
            else:
                continue
            pvalue = results.pvalues.get(name, float("nan"))
            coeffs.append(
                {
                    "event_time_q": int(event_str),
                    "beta": value,
                    "se": results.bse.get(name, float("nan")),
                    "pvalue": pvalue,
                    "term": term,
                }
            )
    coef_df = pd.DataFrame(coeffs)
    metadata = {"n_obs": float(len(df)), "n_firms": float(df["cik"].nunique())}
    return coef_df, metadata
