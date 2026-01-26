"""Census concentration (HHI) ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from industrial_policy.log import get_logger


def ingest_census_concentration(config: Dict[str, Any]) -> Path | None:
    """Ingest user-provided HHI file and write normalized parquet."""
    logger = get_logger()
    hhi_config = config.get("hhi", {})
    if not hhi_config.get("enable", False):
        logger.info("HHI ingestion disabled in config.")
        return None

    source_path = Path(hhi_config.get("census_file", ""))
    if not source_path.exists():
        logger.warning("HHI file not found at %s; skipping.", source_path)
        return None

    naics_col = hhi_config.get("naics_column", "naics")
    hhi_col = hhi_config.get("hhi_column", "hhi")
    df = pd.read_csv(source_path)
    if naics_col not in df.columns or hhi_col not in df.columns:
        logger.warning("HHI file missing required columns; skipping.")
        return None

    df = df[[naics_col, hhi_col]].copy()
    df[naics_col] = df[naics_col].astype(str).str.zfill(2)
    df = df.rename(columns={naics_col: "naics", hhi_col: "hhi"})
    derived_path = Path(config["project"]["data_dir"]) / "derived" / "hhi_by_naics.parquet"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(derived_path, index=False)
    logger.info("Saved HHI parquet to %s", derived_path)
    return derived_path
