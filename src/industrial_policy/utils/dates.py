"""Date utilities."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def parse_date(value: object) -> Optional[pd.Timestamp]:
    """Parse a date-like value to pandas Timestamp.

    Args:
        value: Input value.

    Returns:
        Parsed Timestamp or None.
    """
    if value is None or value == "":
        return None
    return pd.to_datetime(value, errors="coerce")


def quarterize_flow(ytd_values: pd.Series, qtrs: pd.Series) -> pd.Series:
    """Convert YTD flow values to quarterly values.

    Args:
        ytd_values: Series of cumulative values.
        qtrs: Series of quarter counts.

    Returns:
        Series of quarterly flow values.
    """
    quarterly = ytd_values.copy()
    for idx in range(len(ytd_values)):
        if qtrs.iloc[idx] == 1:
            quarterly.iloc[idx] = ytd_values.iloc[idx]
        elif qtrs.iloc[idx] > 1:
            prev_index = idx - 1
            if prev_index >= 0 and qtrs.iloc[prev_index] == qtrs.iloc[idx] - 1:
                quarterly.iloc[idx] = ytd_values.iloc[idx] - ytd_values.iloc[prev_index]
            else:
                quarterly.iloc[idx] = pd.NA
        else:
            quarterly.iloc[idx] = pd.NA
    return quarterly
