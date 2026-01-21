"""Difference-in-differences orchestration."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from industrial_policy.analysis.event_study import build_stacked_dataset, estimate_event_study
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

    results: Dict[str, Dict[str, object]] = {}
    for outcome in outcomes:
        coef_df, meta = estimate_event_study(stacked, outcome, controls)
        results[outcome] = {"coefficients": coef_df, "metadata": meta}

    return results
