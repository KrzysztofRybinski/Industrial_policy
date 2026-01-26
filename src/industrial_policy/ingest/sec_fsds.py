"""SEC Financial Statement Data Sets ingestion."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import duckdb
import pandas as pd
import yaml

from industrial_policy.log import get_logger
from industrial_policy.utils.http import download_file
from industrial_policy.utils.sec import normalize_cik

FLOW_METRICS = {"revenue", "cogs", "gross_profit", "operating_income", "capex_cash", "sga", "rd"}
STOCK_METRICS = {"assets", "ppe_net"}


def _quarters_range(start_year: int, end_year: int) -> Iterable[tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            yield year, q


def _available_columns(conn: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    preview = conn.execute(
        "SELECT * FROM read_csv_auto(?, delim='\t', header=True) LIMIT 0",
        [str(path)],
    ).fetch_df()
    return {col.lower() for col in preview.columns}


def _build_select_exprs(required: Sequence[str], available: set[str]) -> List[str]:
    exprs = []
    for col in required:
        if col in available:
            exprs.append(col)
        else:
            exprs.append(f"NULL AS {col}")
    return exprs


def _load_quarter_df(
    conn: duckdb.DuckDBPyConnection,
    sub_path: Path,
    num_path: Path,
    forms: tuple[str, ...],
    tags_needed: tuple[str, ...],
    keep_dimensions: bool,
) -> pd.DataFrame:
    sub_cols = _available_columns(conn, sub_path)
    num_cols = _available_columns(conn, num_path)

    sub_required = ["adsh", "cik", "accepted", "form", "fy", "fp", "sic"]
    num_required = ["adsh", "tag", "uom", "value", "qtrs", "ddate", "coreg", "segments"]

    sub_exprs = ", ".join(_build_select_exprs(sub_required, sub_cols))
    num_exprs = ", ".join(_build_select_exprs(num_required, num_cols))

    query = (
        "WITH sub AS ("
        "SELECT " + sub_exprs + " FROM read_csv_auto(?, delim='\t', header=True)" "), "
        "num AS ("
        "SELECT " + num_exprs + " FROM read_csv_auto(?, delim='\t', header=True)" "), "
        "filtered AS ("
        "SELECT n.adsh, n.tag, n.uom, n.value, n.qtrs, n.ddate, n.coreg, n.segments, "
        "s.cik, s.accepted, s.form, s.fy, s.fp, s.sic "
        "FROM num n "
        "JOIN sub s ON n.adsh = s.adsh "
        f"WHERE s.form IN {forms} AND n.tag IN {tags_needed} AND n.uom = 'USD'"
    )
    if not keep_dimensions:
        query += " AND n.coreg IS NULL AND (n.segments IS NULL OR n.segments = '')"
    query += ") SELECT * FROM filtered"

    return conn.execute(query, [str(sub_path), str(num_path)]).fetch_df()


def compute_quarterly_from_fsds(df: pd.DataFrame, flow_tags: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["metric_value"] = df["value"].astype("Float64")
    df["quarterly_value"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    flow_mask = df["tag"].isin(flow_tags)
    if not flow_mask.any():
        return df

    for (cik, tag, fy), group in df[flow_mask].groupby(["cik", "tag", "fy"]):
        group = group.sort_values("period_end_date")
        for period_end, period_group in group.groupby("period_end_date"):
            if (period_group["qtrs"] == 1).any():
                for idx in period_group.index:
                    if period_group.loc[idx, "qtrs"] == 1:
                        df.loc[idx, "quarterly_value"] = period_group.loc[idx, "value"]
                    else:
                        df.loc[idx, "quarterly_value"] = pd.NA
                continue

            for idx, row in period_group.iterrows():
                if pd.notna(row["qtrs"]) and row["qtrs"] > 1:
                    prior = group[
                        (group["qtrs"] == row["qtrs"] - 1) &
                        (group["period_end_date"] < period_end)
                    ].sort_values("period_end_date")
                    if not prior.empty:
                        df.loc[idx, "quarterly_value"] = row["value"] - prior.iloc[-1]["value"]
                    else:
                        df.loc[idx, "quarterly_value"] = pd.NA
                else:
                    df.loc[idx, "quarterly_value"] = pd.NA

    df.loc[flow_mask, "metric_value"] = df.loc[flow_mask, "quarterly_value"]
    return df


def build_sec_firm_period_panel(df: pd.DataFrame, tag_map: Dict[str, List[str]]) -> pd.DataFrame:
    tag_priority = {tag: idx for metric_tags in tag_map.values() for idx, tag in enumerate(metric_tags)}
    df = df.copy()
    df["priority"] = df["tag"].map(tag_priority)

    base_cols = [
        "cik",
        "period_end_date",
        "accepted_datetime",
        "fy",
        "fp",
        "form",
        "sic",
    ]
    panel = (
        df.sort_values("accepted_datetime")
        .drop_duplicates(subset=["cik", "period_end_date"], keep="last")[base_cols]
        .sort_values(base_cols)
    )

    for metric, tags in tag_map.items():
        tag_subset = df[df["tag"].isin(tags)].copy()
        if tag_subset.empty:
            panel[metric] = pd.Series(pd.NA, index=panel.index, dtype="Float64")
            continue
        tag_subset = tag_subset.sort_values(["priority", "accepted_datetime"])
        tag_subset = tag_subset.dropna(subset=["metric_value"])
        tag_subset = tag_subset.drop_duplicates(subset=["cik", "period_end_date"], keep="first")
        panel = panel.merge(
            tag_subset[["cik", "period_end_date", "metric_value"]].rename(
                columns={"metric_value": metric}
            ),
            on=["cik", "period_end_date"],
            how="left",
        )
        panel[metric] = panel[metric].astype("Float64")

    if panel.duplicated(subset=["cik", "period_end_date"]).any():
        raise ValueError("SEC firm-period panel contains duplicate cik-period rows.")

    return panel


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_sec_fsds(config: Dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Download and process SEC FSDS data.

    Args:
        config: Loaded configuration.

    Returns:
        Thin panel of firm-period metrics.
    """
    logger = get_logger()
    project = config["project"]
    data_dir = Path(project["data_dir"])
    raw_dir = data_dir / "raw" / "sec_fsds"
    zip_dir = raw_dir / "zips"
    extract_root = raw_dir / "extracted"
    manifest_dir = raw_dir / "manifests"
    zip_dir.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    output_path = derived_dir / "sec_firm_period_base.parquet"
    if output_path.exists() and not force:
        logger.info("SEC panel already exists at %s; skipping download.", output_path)
        return pd.read_parquet(output_path)

    sec_config = config["sec"]
    tag_map_path = Path(sec_config["tags_to_extract"])
    tag_map = yaml.safe_load(tag_map_path.read_text(encoding="utf-8"))
    tags_needed = []
    for metric_tags in tag_map.values():
        for tag in metric_tags:
            if tag not in tags_needed:
                tags_needed.append(tag)

    forms = tuple(sec_config["forms_allowlist"])
    tag_list = tuple(tags_needed)
    flow_tags = {
        tag
        for metric, tags in tag_map.items()
        if metric in FLOW_METRICS
        for tag in tags
    }
    conn = duckdb.connect()

    headers = {"User-Agent": sec_config["user_agent"]}
    quarter_frames: list[pd.DataFrame] = []

    for year, qtr in _quarters_range(sec_config["start_year"], sec_config["end_year"]):
        url = sec_config["base_zip_url"].format(year=year, q=qtr)
        zip_path = zip_dir / f"{year}q{qtr}.zip"
        if not zip_path.exists() or force:
            download_file(url, zip_path, headers=headers, sleep_seconds=sec_config["request_sleep_seconds"])
        else:
            logger.info("SEC zip cached for %sQ%s; skipping download.", year, qtr)

        extract_dir = extract_root / f"{year}q{qtr}"
        if not extract_dir.exists() or force:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extract_dir)
        else:
            logger.info("SEC extraction cached for %sQ%s; skipping extraction.", year, qtr)

        manifest_path = manifest_dir / f"{year}q{qtr}.json"
        manifest = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "zip_path": str(zip_path),
            "zip_size": zip_path.stat().st_size if zip_path.exists() else None,
            "zip_sha256": _sha256(zip_path) if zip_path.exists() else None,
            "extract_dir": str(extract_dir),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        sub_path = extract_dir / "sub.txt"
        num_path = extract_dir / "num.txt"
        if sub_path.exists() and num_path.exists():
            quarter_df = _load_quarter_df(conn, sub_path, num_path, forms, tag_list, sec_config.get("keep_dimensions", False))
            if not quarter_df.empty:
                quarter_frames.append(quarter_df)

    if not quarter_frames:
        logger.warning("No SEC data found after filtering")
        output_path = derived_dir / "sec_firm_period_base.parquet"
        pd.DataFrame().to_parquet(output_path, index=False)
        return pd.DataFrame()

    df = pd.concat(quarter_frames, ignore_index=True)
    df.columns = [col.lower() for col in df.columns]
    df["cik"] = df["cik"].map(normalize_cik)
    df["period_end_date"] = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    df["accepted_datetime"] = pd.to_datetime(df["accepted"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce")

    df = df.sort_values("accepted_datetime")
    df = df.drop_duplicates(subset=["cik", "period_end_date", "tag", "qtrs"], keep="last")

    df = compute_quarterly_from_fsds(df, flow_tags=flow_tags)

    panel = build_sec_firm_period_panel(df, tag_map)
    if "revenue" in panel.columns:
        negative_revenue = panel["revenue"] < 0
        if negative_revenue.any():
            logger.warning("Replacing %s negative revenue values with NaN", negative_revenue.sum())
            panel.loc[negative_revenue, "revenue"] = pd.NA

    panel.to_parquet(output_path, index=False)
    logger.info("Saved SEC panel to %s", output_path)
    return panel
