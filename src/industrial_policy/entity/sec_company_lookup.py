"""SEC company lookup ingestion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from industrial_policy.log import get_logger
from industrial_policy.utils.sec import normalize_cik
from industrial_policy.utils.textnorm import normalize_name

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_company_lookup(data_dir: str, user_agent: Optional[str] = None) -> pd.DataFrame:
    """Download SEC company ticker to CIK mapping.

    Args:
        data_dir: Base data directory.
        user_agent: Optional SEC user agent.

    Returns:
        DataFrame with cik, ticker, company_name, company_name_norm.
    """
    logger = get_logger()
    cache_path = Path(data_dir) / "raw" / "sec_company_tickers.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        headers: Dict[str, str] = {}
        if user_agent:
            headers["User-Agent"] = user_agent
        logger.info("Downloading SEC company tickers")
        response = requests.get(SEC_TICKER_URL, headers=headers, timeout=60)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = []
    for _, entry in data.items():
        rows.append(
            {
                "cik": normalize_cik(entry.get("cik_str")),
                "ticker": entry.get("ticker"),
                "company_name": entry.get("title"),
            }
        )
    df = pd.DataFrame(rows)
    df["company_name_norm"] = df["company_name"].fillna("").map(normalize_name)
    return df
