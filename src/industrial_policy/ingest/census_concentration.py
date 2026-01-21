"""Census concentration ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from industrial_policy.log import get_logger


def load_census_hhi(csv_path: str | Path) -> Optional[pd.DataFrame]:
    """Load user-provided HHI data.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with NAICS and HHI columns if present.
    """
    logger = get_logger()
    path = Path(csv_path)
    if not path.exists():
        logger.info("HHI file not found at %s", path)
        return None

    df = pd.read_csv(path)
    df.columns = [col.lower().strip() for col in df.columns]
    if "naics" not in df.columns:
        raise ValueError("HHI file must contain a 'naics' column")
    if "hhi" not in df.columns:
        raise ValueError("HHI file must contain an 'hhi' column")
    df["naics"] = df["naics"].astype(str)
    return df[["naics", "hhi"]]
