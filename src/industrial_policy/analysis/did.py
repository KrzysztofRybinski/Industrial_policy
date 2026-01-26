"""Difference-in-differences orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from industrial_policy.analysis.event_study import (
    build_stacked_dataset,
    estimate_event_study,
    estimate_event_study_interaction,
)
from industrial_policy.log import get_logger


def run_event_studies(
    treated_panel: pd.DataFrame,
    control_pool: pd.DataFrame,
    matches: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, object]]:
    """Run event-study DID estimation for configured outcomes.

    Args:
        treated_panel: Treated event panel.
        control_pool: Control pool panel.
        matches: Matching table.
        config: Loaded configuration.

    Returns:
        Dictionary of outcome results.
    """
    logger = get_logger()
    window = config["panel"]["event_window_quarters"]
    outcomes: List[str] = config["analysis"]["outcomes"]
    controls = config["analysis"]["control_vars"] if config["analysis"]["add_controls"] else []

    stacked = build_stacked_dataset(treated_panel, control_pool, matches, window)
    if stacked.empty:
        logger.warning("No stacked data available for estimation")
        return {}

    hhi_path = (
        Path(config["project"]["data_dir"]) / "derived" / "hhi_by_naics.parquet"
    )
    if hhi_path.exists() and "naics" in treated_panel.columns:
        hhi = pd.read_parquet(hhi_path)
        stack_naics = treated_panel[["cik", "naics"]].dropna().drop_duplicates()
        stack_naics = stack_naics.rename(columns={"cik": "stack_id"})
        stack_naics["naics"] = stack_naics["naics"].astype(str).str.zfill(2)
        stacked = stacked.merge(stack_naics, on="stack_id", how="left")
        stacked = stacked.merge(hhi, on="naics", how="left")
        high_cut = config.get("analysis", {}).get("heterogeneity", {}).get("high_hhi_cut", 2500)
        stacked["high_hhi"] = (stacked["hhi"] >= high_cut).astype("Int64")
    else:
        logger.info("HHI not available; skipping HHI heterogeneity.")

    results: Dict[str, Dict[str, object]] = {}
    for outcome in outcomes:
        coef_df, meta = estimate_event_study(stacked, outcome, controls)
        payload: Dict[str, object] = {"coefficients": coef_df, "metadata": meta}
        if "high_hhi" in stacked.columns:
            split = {}
            high_df = stacked[stacked["high_hhi"] == 1]
            low_df = stacked[stacked["high_hhi"] == 0]
            split["high"], _ = estimate_event_study(high_df, outcome, controls)
            split["low"], _ = estimate_event_study(low_df, outcome, controls)
            interaction_df, interaction_meta = estimate_event_study_interaction(
                stacked, outcome, controls, "high_hhi"
            )
            payload["hhi_split"] = split
            payload["hhi_interaction"] = {"coefficients": interaction_df, "metadata": interaction_meta}
        results[outcome] = payload

    return results
