"""Optional EU State Aid ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from industrial_policy.log import get_logger
from industrial_policy.utils.textnorm import normalize_name


def load_eu_state_aid(data_dir: str) -> Optional[pd.DataFrame]:
    """Load optional EU State Aid CSV if present.

    Args:
        data_dir: Base data directory.

    Returns:
        DataFrame if present, otherwise None.
    """
    logger = get_logger()
    csv_path = Path(data_dir) / "raw" / "eu_state_aid.csv"
    if not csv_path.exists():
        logger.info("EU State Aid file not found at %s", csv_path)
        return None

    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "beneficiary_name": "beneficiary_name",
            "aid_amount": "aid_amount",
            "aid_date": "aid_date",
            "measure_id": "measure_id",
            "member_state": "member_state",
            "sector_code": "sector_code",
        }
    )
    df["aid_date"] = pd.to_datetime(df["aid_date"], errors="coerce")
    df["beneficiary_name_norm"] = df["beneficiary_name"].fillna("").map(normalize_name)
    return df
