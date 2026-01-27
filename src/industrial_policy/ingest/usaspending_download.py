"""USAspending ingestion via download endpoints.

This uses the async "Download" API (download/count + download/awards + download/status)
to avoid crawling the search endpoint page-by-page.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from industrial_policy.log import get_logger
from industrial_policy.utils.textnorm import normalize_name


class DownloadJobError(RuntimeError):
    """Raised when a USAspending download job fails."""


def _snake_case(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def _hash_payload(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _split_dates(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    days = int((end_date - start_date).days) + 1
    if days <= 1:
        raise ValueError("Cannot split a 1-day interval.")
    mid = start_date + pd.Timedelta(days=(days - 1) // 2)
    left_start = start_date
    left_end = mid
    right_start = mid + pd.Timedelta(days=1)
    right_end = end_date
    if right_start > right_end:  # pragma: no cover
        raise ValueError("Invalid split produced an empty right interval.")
    return left_start, left_end, right_start, right_end


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
        wait=wait_exponential(multiplier=backoff_seconds, min=backoff_seconds, max=30),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            response = session.post(url, json=payload, timeout=timeout)
            if response.status_code in {429} or 500 <= response.status_code < 600:
                response.raise_for_status()
            return response
    raise RuntimeError("USAspending retry failed unexpectedly.")


def _get_with_retry(
    session: requests.Session,
    url: str,
    timeout: int,
    max_attempts: int,
    backoff_seconds: float,
) -> requests.Response:
    retrying = Retrying(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=backoff_seconds, min=backoff_seconds, max=30),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429} or 500 <= response.status_code < 600:
                response.raise_for_status()
            return response
    raise RuntimeError("USAspending retry failed unexpectedly.")


def _build_month_intervals(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.normalize()
    end = end.normalize()
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        interval_end = min(month_end, end)
        intervals.append((cursor, interval_end))
        cursor = interval_end + pd.Timedelta(days=1)
    return intervals


def _normalize_award_id(df: pd.DataFrame) -> pd.Series:
    for col in ("award_id", "award_id_fain", "award_generated_internal_id", "generated_internal_id"):
        if col in df.columns:
            series = df[col].astype("string").fillna("")
            return series.astype(str)
    return pd.Series([""] * len(df))


def _download_count(
    session: requests.Session,
    base_url: str,
    filters: Dict[str, Any],
    request_timeout: int,
    max_attempts: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    url = f"{base_url}/api/v2/download/count/"
    response = _post_with_retry(
        session,
        url,
        {"filters": filters},
        request_timeout,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    response.raise_for_status()
    return response.json()


def _request_download_job(
    session: requests.Session,
    base_url: str,
    payload: Dict[str, Any],
    request_timeout: int,
    max_attempts: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    url = f"{base_url}/api/v2/download/awards/"
    response = _post_with_retry(
        session,
        url,
        payload,
        request_timeout,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    response.raise_for_status()
    return response.json()


def _poll_download_job(
    session: requests.Session,
    status_url: str,
    request_timeout: int,
    poll_seconds: float,
    max_wait_seconds: int,
    max_attempts: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    logger = get_logger()
    start = time.time()
    while True:
        response = _get_with_retry(
            session,
            status_url,
            request_timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        response.raise_for_status()
        status = response.json()
        state = status.get("status")
        if state in {"finished", "failed"}:
            return status
        if time.time() - start > max_wait_seconds:
            raise TimeoutError(f"USAspending download job did not finish within {max_wait_seconds}s.")
        logger.info(
            "USAspending download job running (%ss elapsed); polling again in %ss",
            status.get("seconds_elapsed"),
            poll_seconds,
        )
        time.sleep(poll_seconds)


def _download_file(session: requests.Session, url: str, dest: Path, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _read_prime_awards(zip_path: Path, file_format: str) -> pd.DataFrame:
    suffixes = {".csv", ".tsv", ".txt"}
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in suffixes
        ]
        if not members:
            return pd.DataFrame()
        lowered = {name: name.lower() for name in members}
        prime = [name for name in members if "prime" in lowered[name] and "sub" not in lowered[name]]
        if not prime:
            prime = [name for name in members if "sub" not in lowered[name]]
        selected = prime or members

        sep = ","
        if file_format == "tsv":
            sep = "\t"
        elif file_format == "pstxt":
            sep = "|"

        frames: List[pd.DataFrame] = []
        for name in sorted(selected):
            with archive.open(name) as handle:
                frames.append(pd.read_csv(handle, sep=sep, low_memory=False))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalize_download_df(df: pd.DataFrame, expected_fields: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_snake_case(col) for col in df.columns]
    expected_fields = list(expected_fields)
    for col in expected_fields:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "recipient_name" in df.columns:
        df["recipient_name_norm"] = df["recipient_name"].fillna("").map(normalize_name)
    else:
        df["recipient_name_norm"] = ""
    if "recipient_parent_name" in df.columns:
        df["recipient_parent_name_norm"] = df["recipient_parent_name"].fillna("").map(normalize_name)
    else:
        df["recipient_parent_name_norm"] = ""

    df["award_id"] = _normalize_award_id(df)
    for candidate in ("award_generated_internal_id", "generated_internal_id"):
        if candidate in df.columns:
            df["award_id"] = df["award_id"].mask(
                df["award_id"].eq("") & df[candidate].notna(),
                df[candidate].astype(str),
            )

    recipient_name = (
        df["recipient_name"].astype(str) if "recipient_name" in df.columns else pd.Series("", index=df.index)
    )
    start_date = (
        df["start_date"].astype(str) if "start_date" in df.columns else pd.Series("", index=df.index)
    )
    award_amount = (
        df["award_amount"].astype(str) if "award_amount" in df.columns else pd.Series("", index=df.index)
    )
    fallback_key = df["award_id"].astype(str) + "|" + recipient_name + "|" + start_date + "|" + award_amount
    df["award_key"] = df["award_id"].where(df["award_id"].ne(""), fallback_key)
    df = df.drop_duplicates(subset=["award_key"]).drop(columns=["award_key"])

    schema_cols: List[str] = []
    for col in [
        *expected_fields,
        "award_id",
        "recipient_name_norm",
        "recipient_parent_name_norm",
        "generated_internal_id",
        "award_generated_internal_id",
    ]:
        if col not in schema_cols:
            schema_cols.append(col)
    for col in schema_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[schema_cols]


def _write_parquet_append(chunks: List[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in chunks:
            table = pq.read_table(chunk)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _download_interval(
    session: requests.Session,
    base_url: str,
    filters_template: Dict[str, Any],
    fields: List[str],
    file_format: str,
    request_timeout: int,
    max_attempts: int,
    backoff_seconds: float,
    poll_seconds: float,
    max_wait_seconds: int,
    min_granularity_days: int,
    raw_dir: Path,
    chunks_dir: Path,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    force: bool,
) -> List[Path]:
    logger = get_logger()
    label = f"{interval_start.date().isoformat()}_{interval_end.date().isoformat()}"
    chunk_path = chunks_dir / f"usaspending_download_{label}.parquet"
    manifest_path = chunks_dir / f"usaspending_download_{label}.json"
    zip_path = raw_dir / "downloads" / f"usaspending_download_{label}.zip"

    if chunk_path.exists() and not force:
        logger.info("USAspending download chunk exists; skipping %s", chunk_path)
        return [chunk_path]

    filters = json.loads(json.dumps(filters_template))
    filters["time_period"] = [
        {"start_date": interval_start.date().isoformat(), "end_date": interval_end.date().isoformat()}
    ]
    count = _download_count(
        session,
        base_url,
        filters,
        request_timeout,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    maximum_limit = int(count.get("maximum_limit") or 500000)
    rows = int(count.get("calculated_count") or 0)
    limit = min(rows, maximum_limit)

    logger.info(
        "USAspending download interval %s to %s: %s rows (max %s)",
        interval_start.date().isoformat(),
        interval_end.date().isoformat(),
        rows,
        maximum_limit,
    )

    if rows == 0:
        df = _normalize_download_df(pd.DataFrame(), expected_fields=[_snake_case(c) for c in fields])
        df.to_parquet(chunk_path, index=False)
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interval": {"start_date": interval_start.date().isoformat(), "end_date": interval_end.date().isoformat()},
            "filters": filters,
            "count": count,
            "row_count": 0,
            "stop_reason": "empty",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return [chunk_path]

    if rows > maximum_limit:
        days = int((interval_end - interval_start).days) + 1
        if days <= min_granularity_days:
            diagnostics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Interval exceeds max download limit and cannot be split further.",
                "interval": {"start_date": interval_start.date().isoformat(), "end_date": interval_end.date().isoformat()},
                "count": count,
            }
            manifest_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            raise ValueError(
                f"USAspending download count {rows} exceeds limit {maximum_limit} for a {days}-day interval. "
                "Narrow filters or reduce the time window."
            )
        left_start, left_end, right_start, right_end = _split_dates(interval_start, interval_end)
        logger.info(
            "Splitting USAspending download interval %s to %s (rows %s > max %s)",
            interval_start.date().isoformat(),
            interval_end.date().isoformat(),
            rows,
            maximum_limit,
        )
        return [
            *_download_interval(
                session,
                base_url,
                filters_template,
                fields,
                file_format,
                request_timeout,
                max_attempts,
                backoff_seconds,
                poll_seconds,
                max_wait_seconds,
                min_granularity_days,
                raw_dir,
                chunks_dir,
                left_start,
                left_end,
                force,
            ),
            *_download_interval(
                session,
                base_url,
                filters_template,
                fields,
                file_format,
                request_timeout,
                max_attempts,
                backoff_seconds,
                poll_seconds,
                max_wait_seconds,
                min_granularity_days,
                raw_dir,
                chunks_dir,
                right_start,
                right_end,
                force,
            ),
        ]

    expected_fields = [_snake_case(c) for c in fields]
    request_body = {"filters": filters, "fields": fields, "file_format": file_format, "limit": limit}

    if not zip_path.exists() or force:
        logger.info("Requesting USAspending download job for %s (limit %s)", label, limit)
        job = _request_download_job(
            session,
            base_url,
            request_body,
            request_timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        status_url = job.get("status_url")
        file_url = job.get("file_url")
        if not status_url or not file_url:
            raise DownloadJobError(f"Unexpected download job response: {job}")
        status = _poll_download_job(
            session,
            status_url,
            request_timeout=request_timeout,
            poll_seconds=poll_seconds,
            max_wait_seconds=max_wait_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        if status.get("status") != "finished":
            raise DownloadJobError(f"USAspending download job failed: {status}")
        _download_file(session, file_url, zip_path, timeout=request_timeout)
    else:
        job = {}
        status = {}

    raw_df = _read_prime_awards(zip_path, file_format=file_format)
    cleaned = _normalize_download_df(raw_df, expected_fields=expected_fields)
    cleaned.to_parquet(chunk_path, index=False)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval": {"start_date": interval_start.date().isoformat(), "end_date": interval_end.date().isoformat()},
        "filters": filters,
        "request_hash": _hash_payload(request_body),
        "count": count,
        "job": job,
        "status": status,
        "zip_path": str(zip_path),
        "row_count": int(len(cleaned)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [chunk_path]


def fetch_usaspending_awards_download(config: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Fetch USAspending awards using the Download API and save a parquet panel."""
    logger = get_logger()
    project = config["project"]
    data_dir = Path(project["data_dir"])
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    output_path = derived_dir / "usaspending_awards.parquet"

    api_config = config["usaspending"]
    base_url = api_config["base_url"].rstrip("/")

    request_timeout = int(api_config.get("request_timeout_seconds", 60))
    retry_config = api_config.get("retry", {})
    max_attempts = int(retry_config.get("max_attempts", 8))
    backoff_seconds = float(retry_config.get("backoff_seconds", 1.0))

    download_cfg = api_config.get("download", {})
    poll_seconds = float(download_cfg.get("poll_seconds", 5.0))
    max_wait_seconds = int(download_cfg.get("max_wait_seconds", 7200))
    file_format = str(download_cfg.get("file_format", "csv")).lower()
    min_granularity_days = int(download_cfg.get("min_granularity_days", 1))
    initial_granularity = str(download_cfg.get("initial_granularity", "month")).lower()

    if file_format not in {"csv", "tsv", "pstxt"}:
        raise ValueError("download.file_format must be one of: csv, tsv, pstxt")

    raw_dir = data_dir / "raw" / "usaspending"
    chunks_dir = data_dir / "derived" / "usaspending_download" / "chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    fields = list(api_config.get("fields", []))
    filters_template = api_config.get("filters", {})

    session = requests.Session()
    user_agent = api_config.get("user_agent") or config.get("sec", {}).get("user_agent")
    if user_agent:
        session.headers["User-Agent"] = user_agent
    session.headers.setdefault("Accept", "application/json")

    time_periods = filters_template.get("time_period", [])
    all_chunks: List[Path] = []
    for period in time_periods:
        start = pd.to_datetime(period["start_date"])
        end = pd.to_datetime(period["end_date"])
        if initial_granularity == "month":
            intervals = _build_month_intervals(start, end)
        else:
            # fall back to a single interval; download/count will split if needed
            intervals = [(start, end)]
        for interval_start, interval_end in intervals:
            all_chunks.extend(
                _download_interval(
                    session,
                    base_url,
                    filters_template,
                    fields,
                    file_format,
                    request_timeout,
                    max_attempts,
                    backoff_seconds,
                    poll_seconds,
                    max_wait_seconds,
                    min_granularity_days,
                    raw_dir,
                    chunks_dir,
                    interval_start,
                    interval_end,
                    force,
                )
            )

    seen: set[Path] = set()
    unique_chunks: List[Path] = []
    for chunk in all_chunks:
        if chunk not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk)

    present_chunks = [chunk for chunk in unique_chunks if chunk.exists()]
    missing_chunks = [chunk.name for chunk in unique_chunks if not chunk.exists()]
    if not present_chunks:
        logger.warning("No USAspending download chunks produced.")
        pd.DataFrame().to_parquet(output_path, index=False)
    else:
        logger.info("Assembling %s USAspending chunks into %s", len(present_chunks), output_path)
        _write_parquet_append(present_chunks, output_path)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filters": filters_template,
        "fields": fields,
        "download": {"file_format": file_format, "initial_granularity": initial_granularity},
        "expected_chunks": len(unique_chunks),
        "present_chunks": len(present_chunks),
        "missing_chunks": missing_chunks,
        "output_path": str(output_path),
    }
    manifest_path = Path(project["outputs_dir"]) / "logs" / "usaspending_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if missing_chunks:
        logger.warning(
            "USAspending download ingestion incomplete: %s missing chunks (re-run).", len(missing_chunks)
        )

    return pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
