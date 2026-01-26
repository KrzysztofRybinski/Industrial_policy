"""Event-time panel construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from industrial_policy.log import get_logger
from industrial_policy.utils.sec import normalize_cik


def build_event_panel(
    awards: pd.DataFrame,
    sec_panel: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create treated event-time panel and control pool.

    Args:
        awards: Awards with CIK linkage.
        sec_panel: SEC firm-period panel.
        config: Loaded configuration.

    Returns:
        Tuple of (treated_event_panel, control_pool).
    """
    logger = get_logger()
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    data_dir.mkdir(parents=True, exist_ok=True)

    treated_awards = awards.copy()
    treated_awards["cik"] = treated_awards["cik"].map(normalize_cik)
    treated_awards = treated_awards[treated_awards["treated"] == 1]
    treated_awards = treated_awards.dropna(subset=["cik"])
    treated_awards["award_date"] = pd.to_datetime(treated_awards["award_date"], errors="coerce")

    if config["panel"]["treatment_definition"]["use_first_award_per_firm"]:
        treated_awards = (
            treated_awards.sort_values("award_date").groupby("cik", as_index=False).first()
        )

    sec_panel = sec_panel.copy()
    sec_panel["cik"] = sec_panel["cik"].map(normalize_cik)
    sec_panel["period_end_date"] = pd.to_datetime(sec_panel["period_end_date"], errors="coerce")
    sec_panel["calendar_quarter"] = sec_panel["period_end_date"].dt.to_period("Q")

    award_cols = ["cik", "award_date"]
    if "award_amount" in treated_awards.columns:
        award_cols.append("award_amount")
    if "naics" in treated_awards.columns:
        award_cols.append("naics")
    first_awards = treated_awards[award_cols].copy()
    first_awards = first_awards.rename(
        columns={"award_date": "first_award_date", "award_amount": "first_award_amount"}
    )

    sec_panel = sec_panel.merge(first_awards, on="cik", how="left")

    event_rows = []
    event_lookup = {}
    for _, award in treated_awards.iterrows():
        cik = award["cik"]
        firm_panel = sec_panel[sec_panel["cik"] == cik].sort_values("period_end_date")
        if firm_panel.empty:
            continue
        event_period = firm_panel[firm_panel["period_end_date"] >= award["award_date"]]
        if event_period.empty:
            continue
        event_date = event_period.iloc[0]["period_end_date"]
        event_lookup[cik] = event_date
        firm_panel = firm_panel.copy()
        firm_panel["period_index"] = range(len(firm_panel))
        event_index = firm_panel.loc[
            firm_panel["period_end_date"] == event_date, "period_index"
        ].iloc[0]
        firm_panel["event_time_q"] = firm_panel["period_index"] - event_index
        firm_panel["treated"] = 1
        firm_panel["award_date"] = award["award_date"]
        firm_panel["event_date"] = event_date
        event_rows.append(firm_panel)

    event_panel = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()

    window = config["panel"]["event_window_quarters"]
    if not event_panel.empty:
        event_panel = event_panel[
            (event_panel["event_time_q"] >= -window["pre"])
            & (event_panel["event_time_q"] <= window["post"])
        ]
    if event_panel.empty:
        logger.error(
            "Treated event panel is empty after matching awards to SEC panel. "
            "Check CIK normalization and award dates."
        )
        raise ValueError("No treated event panel rows produced.")

    treated_ciks = treated_awards["cik"].unique().tolist()
    control_pool = sec_panel.copy()
    control_pool["first_award_date"] = pd.to_datetime(control_pool["first_award_date"], errors="coerce")

    stacked_rows = []
    for treated_cik in treated_ciks:
        event_date = event_lookup.get(treated_cik)
        if event_date is None:
            continue
        event_quarter = pd.Period(event_date, freq="Q")
        stack_id = treated_cik
        stack_panel = sec_panel.copy()
        stack_panel = stack_panel[
            stack_panel["calendar_quarter"].between(
                event_quarter - window["pre"], event_quarter + window["post"]
            )
        ]
        stack_panel["event_time_q"] = (
            stack_panel["calendar_quarter"] - event_quarter
        ).astype(int)
        eligible = stack_panel["first_award_date"].isna() | (
            stack_panel["first_award_date"] >= event_date
        )
        stack_panel = stack_panel[eligible].copy()
        later_treated = stack_panel["first_award_date"].notna() & (
            stack_panel["first_award_date"] > event_date
        )
        stack_panel = stack_panel[
            ~later_treated
            | (stack_panel["period_end_date"] < stack_panel["first_award_date"])
        ]
        stack_panel["stack_id"] = stack_id
        stack_panel["treated_in_stack"] = (stack_panel["cik"] == treated_cik).astype(int)
        stack_panel["weight"] = 1.0
        stacked_rows.append(stack_panel)

    stacked_panel = pd.concat(stacked_rows, ignore_index=True) if stacked_rows else pd.DataFrame()
    hhi_path = Path(config["project"]["data_dir"]) / "derived" / "hhi_by_naics.parquet"
    if hhi_path.exists() and not stacked_panel.empty and "naics" in treated_awards.columns:
        hhi = pd.read_parquet(hhi_path)
        stack_naics = (
            treated_awards[["cik", "naics"]]
            .rename(columns={"cik": "stack_id"})
            .dropna(subset=["naics"])
        )
        stack_naics["naics"] = stack_naics["naics"].astype(str).str.zfill(2)
        stacked_panel = stacked_panel.merge(stack_naics, on="stack_id", how="left")
        stacked_panel = stacked_panel.merge(hhi, on="naics", how="left")
        high_cut = config.get("analysis", {}).get("heterogeneity", {}).get("high_hhi_cut", 2500)
        stacked_panel["high_hhi"] = (stacked_panel["hhi"] >= high_cut).astype("Int64")
    else:
        logger.info("HHI not available; skipping HHI join.")
    stacked_path = data_dir / "event_panel_stacked.parquet"
    stacked_panel.to_parquet(stacked_path, index=False)
    logger.info("Saved stacked event panel to %s", stacked_path)

    event_panel_path = data_dir / "event_panel_treated.parquet"
    control_pool_path = data_dir / "panel_pool_controls.parquet"
    event_panel.to_parquet(event_panel_path, index=False)
    control_pool.to_parquet(control_pool_path, index=False)
    logger.info("Saved treated event panel to %s", event_panel_path)
    return event_panel, control_pool
