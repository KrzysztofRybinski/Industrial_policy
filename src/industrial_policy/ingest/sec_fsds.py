"""SEC Financial Statement Data Sets ingestion."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import duckdb
import pandas as pd
import yaml

from industrial_policy.log import get_logger
from industrial_policy.utils.http import download_file

FLOW_METRICS = {"revenue", "cogs", "gross_profit", "operating_income", "capex_cash"}
STOCK_METRICS = {"assets", "ppe_net"}


def _quarters_range(start_year: int, end_year: int) -> Iterable[tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            yield year, q


def _load_txt_to_duckdb(conn: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    logger = get_logger()
    logger.info("Loading %s into %s", path.name, table)
    if conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0] == 0:
        conn.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, delim='\t', header=True)",
            [str(path)],
        )
    else:
        conn.execute(
            f"INSERT INTO {table} SELECT * FROM read_csv_auto(?, delim='\t', header=True)",
            [str(path)],
        )


def _quarterize_flows(df: pd.DataFrame) -> pd.Series:
    df_sorted = df.sort_values(["qtrs", "period_end_date"]).copy()
    values = df_sorted["value"].tolist()
    qtrs = df_sorted["qtrs"].tolist()
    quarterly = []
    for idx, val in enumerate(values):
        if qtrs[idx] == 1:
            quarterly.append(val)
        elif qtrs[idx] > 1:
            prev_idx = idx - 1
            if prev_idx >= 0 and qtrs[prev_idx] == qtrs[idx] - 1:
                quarterly.append(val - values[prev_idx])
            else:
                quarterly.append(pd.NA)
        else:
            quarterly.append(pd.NA)
    df_sorted["quarterly_value"] = quarterly
    return df_sorted.set_index("row_id")["quarterly_value"]


def fetch_sec_fsds(config: Dict[str, Any]) -> pd.DataFrame:
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
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = data_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    sec_config = config["sec"]
    tag_map_path = Path(sec_config["tags_to_extract"])
    tag_map = yaml.safe_load(tag_map_path.read_text(encoding="utf-8"))
    tags_needed = []
    for metric_tags in tag_map.values():
        for tag in metric_tags:
            if tag not in tags_needed:
                tags_needed.append(tag)
    tag_priority = {tag: idx for metric_tags in tag_map.values() for idx, tag in enumerate(metric_tags)}
    tag_to_metric = {tag: metric for metric, tags in tag_map.items() for tag in tags}

    duckdb_path = Path(project["duckdb_path"])
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(duckdb_path))

    headers = {"User-Agent": sec_config["user_agent"]}
    for year, qtr in _quarters_range(sec_config["start_year"], sec_config["end_year"]):
        url = sec_config["base_zip_url"].format(year=year, q=qtr)
        zip_path = raw_dir / f"{year}q{qtr}.zip"
        download_file(url, zip_path, headers=headers, sleep_seconds=sec_config["request_sleep_seconds"])
        extract_dir = raw_dir / f"{year}q{qtr}"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extract_dir)

        sub_path = extract_dir / "sub.txt"
        num_path = extract_dir / "num.txt"
        if sub_path.exists() and num_path.exists():
            _load_txt_to_duckdb(conn, "sec_submissions", sub_path)
            _load_txt_to_duckdb(conn, "sec_numbers", num_path)

    forms = tuple(sec_config["forms_allowlist"])
    tag_list = tuple(tags_needed)

    query = (
        "SELECT n.adsh, n.tag, n.uom, n.value, n.qtrs, n.ddate, n.coreg, n.segments, "
        "s.cik, s.accepted, s.form, s.fy, s.fp, s.sic "
        "FROM sec_numbers n "
        "JOIN sec_submissions s ON n.adsh = s.adsh "
        f"WHERE s.form IN {forms} AND n.tag IN {tag_list}"
    )
    if not sec_config.get("keep_dimensions", False):
        query += " AND n.coreg IS NULL AND (n.segments IS NULL OR n.segments = '')"
    query += " AND n.uom = 'USD'"

    df = conn.execute(query).fetch_df()
    if df.empty:
        logger.warning("No SEC data found after filtering")
        output_path = derived_dir / "sec_firm_period_base.parquet"
        df.to_parquet(output_path, index=False)
        return df

    df.columns = [col.lower() for col in df.columns]
    df["period_end_date"] = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    df["accepted_datetime"] = pd.to_datetime(df["accepted"], errors="coerce")
    df["row_id"] = range(len(df))

    df["metric"] = df["tag"].map(tag_to_metric)
    df["priority"] = df["tag"].map(tag_priority)

    flow_mask = df["metric"].isin(FLOW_METRICS)
    if flow_mask.any():
        df.loc[flow_mask, "quarterly_value"] = (
            df[flow_mask]
            .groupby(["cik", "tag", "fy"], group_keys=False)
            .apply(_quarterize_flows)
        )
    df["metric_value"] = df["value"]
    df.loc[flow_mask, "metric_value"] = df.loc[flow_mask, "quarterly_value"]

    base_cols = [
        "cik",
        "period_end_date",
        "accepted_datetime",
        "fy",
        "fp",
        "form",
        "sic",
    ]
    panel = df[base_cols].drop_duplicates().sort_values(base_cols)

    for metric, tags in tag_map.items():
        tag_subset = df[df["tag"].isin(tags)].copy()
        if tag_subset.empty:
            panel[metric] = pd.NA
            continue
        tag_subset = tag_subset.sort_values(["priority"])  # lower is higher priority
        tag_subset = tag_subset.dropna(subset=["metric_value"])
        tag_subset = tag_subset.drop_duplicates(subset=["cik", "period_end_date"], keep="first")
        panel = panel.merge(
            tag_subset[["cik", "period_end_date", "metric_value"]].rename(
                columns={"metric_value": metric}
            ),
            on=["cik", "period_end_date"],
            how="left",
        )

    output_path = derived_dir / "sec_firm_period_base.parquet"
    panel.to_parquet(output_path, index=False)
    logger.info("Saved SEC panel to %s", output_path)
    return panel
