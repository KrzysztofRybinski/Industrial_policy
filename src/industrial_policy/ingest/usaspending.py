"""USAspending ingestion."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _post_with_retry(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    timeout: int,
) -> requests.Response:
    response = session.post(url, json=payload, timeout=timeout)
    if 500 <= response.status_code < 600:
        response.raise_for_status()
    return response


def _is_page_limit_422(response: requests.Response) -> bool:
    try:
        payload = response.json()
    except ValueError:
        return False
    detail = payload.get("detail") or payload.get("errors") or payload.get("message")
    if isinstance(detail, str):
        detail_text = detail.lower()
        return "page" in detail_text and (
            "limit" in detail_text
            or "max" in detail_text
            or "out of range" in detail_text
            or "less than" in detail_text
        )
    if isinstance(detail, list):
        for item in detail:
            if isinstance(item, dict):
                loc = item.get("loc", [])
                if isinstance(loc, (list, tuple)) and any(
                    str(part).lower() == "page" for part in loc
                ):
                    return True
                msg = str(item.get("msg", "")).lower()
                if "page" in msg and (
                    "limit" in msg or "max" in msg or "out of range" in msg
                ):
                    return True
    return False


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
    rows: List[Dict[str, Any]] = []
    seen_signatures: Set[tuple] = set()
    stop_reason: Optional[str] = None
    max_pages = api_config.get("max_pages", 99999)
    max_records = api_config.get("max_records")
    request_timeout = api_config.get("request_timeout_seconds", 60)
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
        try:
            response = _post_with_retry(session, url, payload, request_timeout)
        except requests.RequestException as exc:
            stop_reason = "request_error"
            status = getattr(exc.response, "status_code", None)
            logger.error(
                "USAspending request failed on page %s (status=%s): %s",
                page,
                status,
                exc,
            )
            break
        if response.status_code == 422:
            if _is_page_limit_422(response):
                stop_reason = "api_page_limit"
                logger.warning(
                    "USAspending returned page-limit 422 for page %s; stopping ingestion",
                    page,
                )
                break
            logger.error(
                "USAspending returned 422 for page %s: %s",
                page,
                response.text,
            )
            response.raise_for_status()
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            stop_reason = "empty_page"
            break
        signature = tuple(sorted(str(item.get("Award ID", "")) for item in results))
        if signature in seen_signatures:
            stop_reason = "repeated_page"
            break
        seen_signatures.add(signature)
        rows.extend(results)
        logger.info(
            "USAspending page %s returned %s rows (cumulative %s)",
            page,
            len(results),
            len(rows),
        )
        if max_records is not None and len(rows) >= max_records:
            stop_reason = "max_records"
            rows = rows[:max_records]
            break
        page += 1
        if page > max_pages:
            stop_reason = "max_pages"
            break

    if stop_reason:
        logger.info("Stopping USAspending ingestion due to %s", stop_reason)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No USAspending awards returned")
    else:
        df.columns = [_snake_case(col) for col in df.columns]
        date_cols = ["start_date", "end_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "recipient_name" in df.columns:
            df["recipient_name_norm"] = df["recipient_name"].fillna("").map(normalize_name)

    df.to_parquet(output_path, index=False)
    logger.info("Saved awards to %s", output_path)
    pages_pulled = len(seen_signatures)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filters": payload["filters"],
        "fields": payload["fields"],
        "page_size": payload["limit"],
        "subawards": payload["subawards"],
        "request_timeout_seconds": request_timeout,
        "total_pages": pages_pulled,
        "total_rows": len(df),
        "stop_reason": stop_reason,
    }
    manifest_path = Path(project["outputs_dir"]) / "logs" / "usaspending_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved USAspending manifest to %s", manifest_path)
    return df
