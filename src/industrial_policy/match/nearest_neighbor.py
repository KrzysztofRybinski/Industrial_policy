"""Nearest-neighbor matching."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from industrial_policy.log import get_logger


def nearest_neighbor_match(propensity: pd.DataFrame, config: Dict[str, Any], k: int = 3, caliper: float = 0.05) -> pd.DataFrame:
    """Match treated firms to nearest controls by propensity score.

    Args:
        propensity: Propensity score dataframe.
        config: Loaded configuration.
        k: Number of controls per treated.
        caliper: Maximum allowed propensity distance.

    Returns:
        DataFrame of matches.
    """
    logger = get_logger()
    matches = []

    treated = propensity[propensity["treated"] == 1].copy()
    controls = propensity[propensity["treated"] == 0].copy()

    treated["sic2"] = treated["sic"].astype(str).str[:2]
    controls["sic2"] = controls["sic"].astype(str).str[:2]

    for _, treat_row in treated.iterrows():
        year = treat_row["event_year"]
        candidates = controls[
            (controls["sic2"] == treat_row["sic2"]) &
            (controls["event_year"].between(year - 1, year + 1))
        ].copy()
        if candidates.empty:
            continue
        candidates["distance"] = (candidates["propensity_score"] - treat_row["propensity_score"]).abs()
        candidates = candidates[candidates["distance"] <= caliper]
        candidates = candidates.nsmallest(k, "distance")
        if candidates.empty:
            continue
        weight = 1 / len(candidates)
        for _, control_row in candidates.iterrows():
            matches.append(
                {
                    "treated_cik": treat_row["cik"],
                    "control_cik": control_row["cik"],
                    "treated_year": year,
                    "control_year": control_row["event_year"],
                    "weight": weight,
                    "distance": control_row["distance"],
                }
            )

    matches_df = pd.DataFrame(matches)
    output_path = Path(config["project"]["data_dir"]) / "derived" / "matches.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_parquet(output_path, index=False)
    logger.info("Saved matches to %s", output_path)
    return matches_df
