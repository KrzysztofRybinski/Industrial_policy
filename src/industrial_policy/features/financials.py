"""Financial feature engineering."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize a series at specified percentiles."""
    if series.empty:
        return series
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower=lower_val, upper=upper_val)


def compute_financial_features(df: pd.DataFrame, winsor_limits: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Compute markup and capacity proxies.

    Args:
        df: SEC firm-period base panel.
        winsor_limits: Lower and upper winsorization percentiles.

    Returns:
        DataFrame with new features.
    """
    df = df.copy()
    df["gross_margin"] = (df["revenue"] - df["cogs"]) / df["revenue"]
    df["markup_rev_cogs"] = df["revenue"] / df["cogs"]
    df["capex"] = df["capex_cash"].abs()
    df["capex_intensity"] = df["capex"] / df["revenue"]

    df = df.sort_values(["cik", "period_end_date"]).copy()
    df["ppe_growth"] = np.log(df["ppe_net"]) - np.log(df.groupby("cik")["ppe_net"].shift(1))
    df["log_sales"] = np.log(df["revenue"])
    df["log_assets"] = np.log(df["assets"])

    ratio_cols: Iterable[str] = [
        "gross_margin",
        "markup_rev_cogs",
        "capex_intensity",
        "ppe_growth",
    ]
    for col in ratio_cols:
        df[f"{col}_winsor"] = winsorize(df[col], *winsor_limits)

    return df
