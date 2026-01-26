"""Output writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from industrial_policy.log import get_logger


def save_event_study_outputs(results: Dict[str, Dict[str, object]], outputs_dir: str) -> None:
    """Save event-study coefficient tables and metadata.

    Args:
        results: Output of run_event_studies.
        outputs_dir: Base outputs directory.
    """
    logger = get_logger()
    outputs_path = Path(outputs_dir)
    tables_dir = outputs_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for outcome, payload in results.items():
        coef_df: pd.DataFrame = payload["coefficients"]
        meta = payload["metadata"]

        csv_path = tables_dir / f"event_study_{outcome}.csv"
        meta_path = tables_dir / f"event_study_{outcome}_meta.json"

        coef_df.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Saved outputs for %s", outcome)

        hhi_split = payload.get("hhi_split")
        if isinstance(hhi_split, dict):
            high_df = hhi_split.get("high", pd.DataFrame())
            low_df = hhi_split.get("low", pd.DataFrame())
            split_path = tables_dir / f"event_study_{outcome}_hhi_split.csv"
            combined = []
            if not high_df.empty:
                high_df = high_df.copy()
                high_df["group"] = "high"
                combined.append(high_df)
            if not low_df.empty:
                low_df = low_df.copy()
                low_df["group"] = "low"
                combined.append(low_df)
            if combined:
                pd.concat(combined, ignore_index=True).to_csv(split_path, index=False)
        hhi_interaction = payload.get("hhi_interaction", {})
        if isinstance(hhi_interaction, dict):
            interaction_df = hhi_interaction.get("coefficients", pd.DataFrame())
            if isinstance(interaction_df, pd.DataFrame) and not interaction_df.empty:
                interaction_path = tables_dir / f"event_study_{outcome}_hhi_interaction.csv"
                interaction_df.to_csv(interaction_path, index=False)


def save_event_study_summary(results: Dict[str, Dict[str, object]], outputs_dir: str) -> None:
    """Save horizon summaries across outcomes."""
    summaries = []
    for outcome, payload in results.items():
        coef_df: pd.DataFrame = payload["coefficients"]
        if coef_df.empty:
            continue
        for start, end, label in [(0, 2, "0_2"), (3, 6, "3_6"), (7, 12, "7_12")]:
            subset = coef_df[
                (coef_df["event_time_q"] >= start) & (coef_df["event_time_q"] <= end)
            ]
            if subset.empty:
                avg = float("nan")
            else:
                avg = float(subset["beta"].mean())
            summaries.append({"outcome": outcome, "horizon": label, "avg_effect": avg})
    if summaries:
        summary_df = pd.DataFrame(summaries)
        tables_dir = Path(outputs_dir) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_path = tables_dir / "event_study_summary.csv"
        summary_df.to_csv(summary_path, index=False)
