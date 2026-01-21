"""Sample construction reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from industrial_policy.log import get_logger


def _count_unique(series: Optional[pd.Series]) -> Optional[int]:
    if series is None:
        return None
    return int(series.dropna().nunique())


def write_sample_construction_report(
    config: Dict[str, Any],
    outputs_dir: str | Path,
    logger=None,
) -> None:
    """Write sample construction report to CSV.

    Args:
        config: Loaded configuration.
        outputs_dir: Base outputs directory.
    """
    if logger is None:
        logger = get_logger()
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    tables_dir = Path(outputs_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    awards_path = data_dir / "usaspending_awards.parquet"
    linked_awards_path = data_dir / "awards_with_cik.parquet"
    sec_features_path = data_dir / "sec_firm_period_features.parquet"
    treated_panel_path = data_dir / "event_panel_treated.parquet"
    matches_path = data_dir / "matches.parquet"

    awards = pd.read_parquet(awards_path) if awards_path.exists() else None
    linked_awards = (
        pd.read_parquet(linked_awards_path) if linked_awards_path.exists() else None
    )
    sec_features = pd.read_parquet(sec_features_path) if sec_features_path.exists() else None
    treated_panel = (
        pd.read_parquet(treated_panel_path) if treated_panel_path.exists() else None
    )
    matches = pd.read_parquet(matches_path) if matches_path.exists() else None

    report_rows: List[Dict[str, Optional[float]]] = []

    report_rows.append(
        {
            "metric": "awards_downloaded_rows",
            "value": float(len(awards)) if awards is not None else None,
        }
    )

    min_award = config["panel"]["treatment_definition"]["min_award_amount"]
    awards_for_counts = linked_awards if linked_awards is not None else awards
    if awards_for_counts is not None and "treated" in awards_for_counts.columns:
        treated_awards = awards_for_counts[awards_for_counts["treated"] == 1]
        treated_count = len(treated_awards)
    else:
        treated_count = None
    report_rows.append(
        {
            "metric": f"awards_passing_min_award_amount_{min_award}",
            "value": float(treated_count) if treated_count is not None else None,
        }
    )

    if awards_for_counts is not None:
        recipient_series = (
            awards_for_counts["recipient_name_norm"]
            if "recipient_name_norm" in awards_for_counts.columns
            else awards_for_counts.get("recipient_name")
        )
        recipients = _count_unique(recipient_series)
    else:
        recipients = None
    report_rows.append({"metric": "unique_recipients", "value": recipients})

    if linked_awards is not None and "cik" in linked_awards.columns:
        matched_ciks = _count_unique(linked_awards["cik"])
    else:
        matched_ciks = None
    report_rows.append({"metric": "unique_cik_matched", "value": matched_ciks})

    if linked_awards is not None and sec_features is not None:
        linked_ciks = linked_awards["cik"].dropna().unique()
        sec_ciks = sec_features["cik"].dropna().unique()
        merged_ciks = len(set(linked_ciks).intersection(sec_ciks))
    else:
        merged_ciks = None
    report_rows.append({"metric": "unique_cik_in_sec_panel", "value": merged_ciks})

    baseline = config["panel"]["baseline_quarters"]
    if treated_panel is not None and "event_time_q" in treated_panel.columns:
        baseline_panel = treated_panel[
            (treated_panel["event_time_q"] >= baseline["start"]) &
            (treated_panel["event_time_q"] <= baseline["end"])
        ]
        baseline_ciks = _count_unique(baseline_panel["cik"])
    else:
        baseline_ciks = None
    report_rows.append(
        {"metric": "treated_firms_with_baseline_window", "value": baseline_ciks}
    )

    if matches is not None and "treated_cik" in matches.columns:
        matched_treated = _count_unique(matches["treated_cik"])
    else:
        matched_treated = None
    report_rows.append({"metric": "matched_treated_firms", "value": matched_treated})

    hhi_config = config.get("analysis", {}).get("heterogeneity", {})
    if hhi_config.get("enable_hhi") and treated_panel is not None:
        hhi_coverage = None
        if "hhi" in treated_panel.columns:
            treated_unique = treated_panel["cik"].dropna().unique()
            hhi_unique = treated_panel.loc[treated_panel["hhi"].notna(), "cik"].unique()
            hhi_coverage = (
                float(len(hhi_unique)) / float(len(treated_unique)) * 100
                if len(treated_unique) > 0
                else None
            )
        report_rows.append(
            {"metric": "treated_firms_with_hhi_coverage_pct", "value": hhi_coverage}
        )

    report_df = pd.DataFrame(report_rows)
    output_path = tables_dir / "sample_construction.csv"
    report_df.to_csv(output_path, index=False)
    logger.info("Saved sample construction report to %s", output_path)
