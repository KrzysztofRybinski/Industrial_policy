"""Award features."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from industrial_policy.utils.textnorm import normalize_name


def prepare_awards(awards: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Clean award data and define treatment flags.

    Args:
        awards: Raw awards dataframe.
        config: Loaded configuration.

    Returns:
        Cleaned awards dataframe.
    """
    awards = awards.copy()
    awards["recipient_name_norm"] = awards["recipient_name"].fillna("").map(normalize_name)

    awards["award_date"] = awards["start_date"].fillna(awards["end_date"])
    awards["award_amount"] = pd.to_numeric(awards["award_amount"], errors="coerce")
    min_award = config["panel"]["treatment_definition"]["min_award_amount"]
    awards["treated"] = (awards["award_amount"] >= min_award).astype(int)
    return awards
