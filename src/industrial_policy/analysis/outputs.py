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
        tex_path = tables_dir / f"event_study_{outcome}.tex"
        meta_path = tables_dir / f"event_study_{outcome}_meta.json"

        coef_df.to_csv(csv_path, index=False)
        coef_df.to_latex(tex_path, index=False, float_format="%.4f")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Saved outputs for %s", outcome)
