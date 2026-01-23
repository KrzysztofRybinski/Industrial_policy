"""SEC-specific helpers."""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def normalize_cik(value: object) -> Optional[str]:
    """Normalize a CIK to a zero-padded 10-character string."""
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() == "nan":
        return None
    raw = re.sub(r"\.0+$", "", raw)
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return digits.zfill(10)
