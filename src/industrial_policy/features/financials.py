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
    required_cols = ["revenue", "cogs", "capex_cash", "ppe_net", "assets"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    if "sga" not in df.columns:
        df["sga"] = np.nan
    if "rd" not in df.columns:
        df["rd"] = np.nan

    df["gross_margin"] = np.where(
        df["revenue"] > 0, (df["revenue"] - df["cogs"]) / df["revenue"], np.nan
    )
    df["markup_rev_cogs"] = np.where(
        df["cogs"] > 0, df["revenue"] / df["cogs"], np.nan
    )
    df["capex"] = df["capex_cash"].abs()
    df["capex_intensity"] = np.where(
        df["revenue"] > 0, df["capex"] / df["revenue"], np.nan
    )
    df["sga_intensity"] = np.where(
        df["revenue"] > 0, df["sga"] / df["revenue"], np.nan
    )
    df["rd_intensity"] = np.where(
        df["revenue"] > 0, df["rd"] / df["revenue"], np.nan
    )

    df = df.sort_values(["cik", "period_end_date"]).copy()
    log_ppe = np.where(df["ppe_net"] > 0, np.log(df["ppe_net"]), np.nan)
    df["ppe_growth"] = log_ppe - pd.Series(log_ppe).groupby(df["cik"]).shift(1)
    df["log_sales"] = np.where(df["revenue"] > 0, np.log(df["revenue"]), np.nan)
    df["log_assets"] = np.where(df["assets"] > 0, np.log(df["assets"]), np.nan)

    ratio_cols: Iterable[str] = [
        "gross_margin",
        "markup_rev_cogs",
        "capex_intensity",
        "ppe_growth",
        "sga_intensity",
        "rd_intensity",
    ]
    for col in ratio_cols:
        df[f"{col}_winsor"] = winsorize(df[col], *winsor_limits)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
