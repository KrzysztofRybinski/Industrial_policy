"""USAspending ingestion."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

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


class PageLimitError(RuntimeError):
    """Raised when a USAspending query exceeds page limits."""


def _post_with_retry(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    timeout: int,
    max_attempts: int,
    backoff_seconds: float,
) -> requests.Response:
    retrying = Retrying(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=backoff_seconds, min=backoff_seconds, max=10),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            response = session.post(url, json=payload, timeout=timeout)
            if 500 <= response.status_code < 600:
                response.raise_for_status()
            return response
    raise RuntimeError("USAspending retry failed unexpectedly.")


def _interval_label(start: datetime, end: datetime) -> str:
    return f"{start.date().isoformat()}_{end.date().isoformat()}"


def _hash_payload(payload: Dict[str, Any]) -> str:
    return str(abs(hash(json.dumps(payload, sort_keys=True))))


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


def _build_intervals(
    start: datetime,
    end: datetime,
    granularity: str,
) -> List[Tuple[datetime, datetime]]:
    intervals: List[Tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        if granularity == "month":
            next_cursor = (cursor.replace(day=1) + pd.offsets.MonthEnd(1)).to_pydatetime()
        elif granularity == "week":
            next_cursor = (cursor + pd.Timedelta(days=6)).to_pydatetime()
        else:  # quarter
            quarter_end = (cursor + pd.offsets.QuarterEnd(0)).to_pydatetime()
            next_cursor = quarter_end
        interval_end = min(next_cursor, end)
        intervals.append((cursor, interval_end))
        cursor = interval_end + pd.Timedelta(days=1)
    return intervals


def _normalize_award_id(df: pd.DataFrame) -> pd.Series:
    for col in ("award_id", "award_id_fain", "award_generated_internal_id"):
        if col in df.columns:
            return df[col].astype(str).fillna("")
    return pd.Series([""] * len(df))


def _fetch_interval(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    request_timeout: int,
    max_pages: int,
    adaptive_split_on_422: bool,
    max_attempts: int,
    backoff_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger = get_logger()
    page = 1
    rows: List[Dict[str, Any]] = []
    stop_reason: Optional[str] = None
    total_pages: Optional[int] = None

    while True:
        page_payload = {**payload, "page": page}
        logger.info(
            "USAspending interval %s to %s page %s",
            payload["filters"]["time_period"][0]["start_date"],
            payload["filters"]["time_period"][0]["end_date"],
            page,
        )
        response = _post_with_retry(
            session,
            url,
            page_payload,
            request_timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        if response.status_code == 422 and adaptive_split_on_422 and _is_page_limit_422(response):
            raise PageLimitError("API page limit hit (422).")
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            stop_reason = "empty_page"
            break
        rows.extend(results)
        page_meta = data.get("page_metadata", {})
        if total_pages is None:
            total_pages = page_meta.get("total_pages")
        if page >= max_pages:
            raise PageLimitError("Interval exceeds max_pages_per_query.")
        if total_pages is not None and page >= total_pages:
            stop_reason = "total_pages"
            break
        page += 1

    df = pd.DataFrame(rows)
    return df, {
        "pages_downloaded": page,
        "total_pages": total_pages,
        "stop_reason": stop_reason,
    }


def _write_chunk(
    df: pd.DataFrame,
    chunk_path: Path,
    manifest_path: Path,
    payload: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    df.to_parquet(chunk_path, index=False)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_body": payload,
        "request_hash": _hash_payload(payload),
        "row_count": len(df),
        **meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _download_interval(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    request_timeout: int,
    max_pages: int,
    adaptive_split_on_422: bool,
    max_attempts: int,
    backoff_seconds: float,
    min_granularity_days: int,
    chunk_path: Path,
    manifest_path: Path,
    force: bool,
) -> None:
    logger = get_logger()
    if chunk_path.exists() and not force:
        logger.info("USAspending chunk exists; skipping %s", chunk_path)
        return
    try:
        df, meta = _fetch_interval(
            session,
            url,
            payload,
            request_timeout,
            max_pages,
            adaptive_split_on_422,
            max_attempts,
            backoff_seconds,
        )
        _write_chunk(df, chunk_path, manifest_path, payload, meta)
    except PageLimitError:
        start_date = pd.to_datetime(payload["filters"]["time_period"][0]["start_date"])
        end_date = pd.to_datetime(payload["filters"]["time_period"][0]["end_date"])
        days = (end_date - start_date).days + 1
        if days <= min_granularity_days:
            diagnostics = {
                "error": "Interval cannot be split further.",
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
            }
            manifest_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            raise
        mid = start_date + pd.Timedelta(days=math.floor(days / 2))
        left = payload.copy()
        right = payload.copy()
        left["filters"] = json.loads(json.dumps(payload["filters"]))
        right["filters"] = json.loads(json.dumps(payload["filters"]))
        left["filters"]["time_period"][0]["start_date"] = start_date.date().isoformat()
        left["filters"]["time_period"][0]["end_date"] = mid.date().isoformat()
        right["filters"]["time_period"][0]["start_date"] = (mid + pd.Timedelta(days=1)).date().isoformat()
        right["filters"]["time_period"][0]["end_date"] = end_date.date().isoformat()
        left_chunk = chunk_path.with_name(f"usaspending_{_interval_label(start_date, mid)}.parquet")
        right_chunk = chunk_path.with_name(
            f"usaspending_{_interval_label(mid + pd.Timedelta(days=1), end_date)}.parquet"
        )
        left_manifest = left_chunk.with_suffix(".json")
        right_manifest = right_chunk.with_suffix(".json")
        _download_interval(
            session,
            url,
            left,
            request_timeout,
            max_pages,
            adaptive_split_on_422,
            max_attempts,
            backoff_seconds,
            min_granularity_days,
            left_chunk,
            left_manifest,
            force,
        )
        _download_interval(
            session,
            url,
            right,
            request_timeout,
            max_pages,
            adaptive_split_on_422,
            max_attempts,
            backoff_seconds,
            min_granularity_days,
            right_chunk,
            right_manifest,
            force,
        )


def fetch_usaspending_awards(config: Dict[str, Any], force: bool = False) -> pd.DataFrame:
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

    request_timeout = api_config.get("request_timeout_seconds", 60)
    max_pages = api_config.get("max_pages_per_query", 450)
    chunk_config = api_config.get("chunking", {})
    min_granularity_days = int(chunk_config.get("min_granularity_days", 1))
    adaptive_split_on_422 = bool(chunk_config.get("adaptive_split_on_422", True))
    force = force or api_config.get("force_refresh", False)
    retry_config = api_config.get("retry", {})
    max_attempts = int(retry_config.get("max_attempts", 8))
    backoff_seconds = float(retry_config.get("backoff_seconds", 1.0))
    session = requests.Session()

    chunks_dir = data_dir / "usaspending" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    time_periods = api_config["filters"]["time_period"]
    initial_granularity = chunk_config.get("initial_granularity", "quarter")
    payload_template = {
        "filters": api_config["filters"],
        "fields": api_config["fields"],
        "limit": api_config.get("page_size", 100),
        "subawards": api_config.get("subawards", False),
    }

    all_chunks: List[Path] = []
    for period in time_periods:
        start = pd.to_datetime(period["start_date"])
        end = pd.to_datetime(period["end_date"])
        intervals = _build_intervals(start, end, initial_granularity)
        for interval_start, interval_end in intervals:
            payload = json.loads(json.dumps(payload_template))
            payload["filters"]["time_period"] = [
                {
                    "start_date": interval_start.date().isoformat(),
                    "end_date": interval_end.date().isoformat(),
                }
            ]
            label = _interval_label(interval_start, interval_end)
            chunk_path = chunks_dir / f"usaspending_{label}.parquet"
            manifest_path = chunks_dir / f"usaspending_{label}.json"
            _download_interval(
                session,
                url,
                payload,
                request_timeout,
                max_pages,
                adaptive_split_on_422,
                max_attempts,
                backoff_seconds,
                min_granularity_days,
                chunk_path,
                manifest_path,
                force,
            )
            all_chunks.append(chunk_path)

    frames = []
    for chunk_path in all_chunks:
        if chunk_path.exists():
            frames.append(pd.read_parquet(chunk_path))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if df.empty:
        logger.warning("No USAspending awards returned after chunk assembly")
    else:
        df.columns = [_snake_case(col) for col in df.columns]
        expected_fields = {_snake_case(field) for field in api_config.get("fields", [])}
        missing_fields = expected_fields.difference(df.columns)
        for field in sorted(missing_fields):
            logger.warning("USAspending field not returned by API: %s", field)
        date_cols = ["start_date", "end_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "recipient_name" in df.columns:
            df["recipient_name_norm"] = df["recipient_name"].fillna("").map(normalize_name)
        if "recipient_parent_name" in df.columns:
            df["recipient_parent_name_norm"] = (
                df["recipient_parent_name"].fillna("").map(normalize_name)
            )

        df["award_id"] = _normalize_award_id(df)
        if "award_generated_internal_id" in df.columns:
            df["award_id"] = df["award_id"].mask(
                df["award_id"].eq("") & df["award_generated_internal_id"].notna(),
                df["award_generated_internal_id"].astype(str),
            )
        df["award_key"] = df["award_id"].where(
            df["award_id"].ne(""),
            df["award_id"]
            + "|"
            + df.get("recipient_name", "").astype(str)
            + "|"
            + df.get("start_date", "").astype(str)
            + "|"
            + df.get("award_amount", "").astype(str),
        )
        before = len(df)
        df = df.drop_duplicates(subset=["award_key"])
        removed = before - len(df)
        logger.info("Removed %s duplicate awards after chunk assembly", removed)
        df = df.drop(columns=["award_key"])

    df.to_parquet(output_path, index=False)
    logger.info("Saved awards to %s", output_path)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filters": api_config["filters"],
        "fields": api_config["fields"],
        "page_size": api_config.get("page_size", 100),
        "total_rows": len(df),
        "chunks": [chunk.name for chunk in all_chunks],
    }
    manifest_path = Path(project["outputs_dir"]) / "logs" / "usaspending_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved USAspending manifest to %s", manifest_path)
    return df
