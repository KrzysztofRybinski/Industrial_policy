"""USAspending ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from industrial_policy.log import get_logger
from industrial_policy.utils.textnorm import normalize_name


def _snake_case(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def fetch_usaspending_awards(config: Dict[str, Any]) -> pd.DataFrame:
    """Fetch awards from the USAspending API and save to parquet.

    Args:
        config: Loaded configuration.

    Returns:
        DataFrame of normalized awards.
    """
    logger = get_logger()
    project = config["project"]
    data_dir = Path(project["data_dir"]) / "derived"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "usaspending_awards.parquet"

    api_config = config["usaspending"]
    base_url = api_config["base_url"].rstrip("/")
    endpoint = api_config["endpoint"].lstrip("/")
    url = f"{base_url}/{endpoint}"

    page = 1
    rows = []
    session = requests.Session()
    while True:
        payload = {
            "filters": api_config["filters"],
            "fields": api_config["fields"],
            "page": page,
            "limit": api_config.get("page_size", 100),
            "subawards": api_config.get("subawards", False),
        }
        logger.info("Fetching USAspending page %s", page)
        response = session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            break
        rows.extend(results)
        page += 1
        if page > api_config.get("max_pages", 99999):
            break

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No USAspending awards returned")
        df.to_parquet(output_path, index=False)
        return df

    df.columns = [_snake_case(col) for col in df.columns]
    date_cols = ["start_date", "end_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "recipient_name" in df.columns:
        df["recipient_name_norm"] = df["recipient_name"].fillna("").map(normalize_name)

    df.to_parquet(output_path, index=False)
    logger.info("Saved awards to %s", output_path)
    return df
