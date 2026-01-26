"""Robustness suite runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from industrial_policy.analysis.did import run_event_studies
from industrial_policy.analysis.outputs import save_event_study_outputs
from industrial_policy.log import get_logger


def _apply_exclusions(df: pd.DataFrame, exclude_periods: list[dict[str, str]]) -> pd.DataFrame:
    if df.empty or not exclude_periods:
        return df
    df = df.copy()
    df["period_end_date"] = pd.to_datetime(df["period_end_date"], errors="coerce")
    for period in exclude_periods:
        start = pd.to_datetime(period["start"])
        end = pd.to_datetime(period["end"])
        df = df[~df["period_end_date"].between(start, end)]
    return df


def run_robustness_suite(config: Dict[str, Any]) -> None:
    """Run robustness checks over multiple specs."""
    logger = get_logger()
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    outputs_root = Path(config["project"]["outputs_dir"])
    tables_root = outputs_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    outputs_dir = tables_root / "robustness"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    treated_path = data_dir / "event_panel_treated.parquet"
    control_path = data_dir / "panel_pool_controls.parquet"
    matches_path = data_dir / "matches.parquet"
    if not (treated_path.exists() and control_path.exists() and matches_path.exists()):
        logger.warning("Missing inputs for robustness; run build panel and match controls.")
        return

    treated_panel = pd.read_parquet(treated_path)
    control_pool = pd.read_parquet(control_path)
    matches = pd.read_parquet(matches_path)

    robustness_cfg = config.get("analysis", {}).get("robustness", {})
    windows = robustness_cfg.get("pre_post_windows", [])
    exclude_periods = robustness_cfg.get("exclude_periods", [])

    summary_rows = []
    for idx, window in enumerate(windows, start=1):
        spec_config = {**config}
        spec_config["panel"] = {**config["panel"], "event_window_quarters": window}
        treated_spec = _apply_exclusions(treated_panel, exclude_periods)
        control_spec = _apply_exclusions(control_pool, exclude_periods)
        results = run_event_studies(treated_spec, control_spec, matches, spec_config)
        spec_dir = outputs_dir / f"spec_{idx}"
        spec_dir.mkdir(parents=True, exist_ok=True)
        save_event_study_outputs(results, str(spec_dir))
        summary_rows.append(
            {
                "spec_id": idx,
                "pre": window["pre"],
                "post": window["post"],
                "exclude_periods": bool(exclude_periods),
            }
        )

    if summary_rows:
        summary_path = tables_root / "robustness_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info("Saved robustness summary to %s", summary_path)
