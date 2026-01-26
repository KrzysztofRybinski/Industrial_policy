from __future__ import annotations

import pandas as pd

from industrial_policy.features.panel import build_event_panel


def test_riskset_censoring(tmp_path):
    awards = pd.DataFrame(
        {
            "cik": ["0000000001", "0000000002"],
            "treated": [1, 1],
            "award_date": [pd.Timestamp("2020-04-15"), pd.Timestamp("2020-11-15")],
            "award_amount": [100.0, 200.0],
            "naics": ["11", "11"],
        }
    )
    sec_panel = pd.DataFrame(
        {
            "cik": ["0000000001"] * 4 + ["0000000002"] * 4,
            "period_end_date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-06-30",
                    "2020-09-30",
                    "2020-12-31",
                    "2020-03-31",
                    "2020-06-30",
                    "2020-09-30",
                    "2020-12-31",
                ]
            ),
            "revenue": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    config = {
        "project": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path / "outputs")},
        "panel": {"event_window_quarters": {"pre": 1, "post": 2}, "treatment_definition": {"use_first_award_per_firm": True}},
    }
    build_event_panel(awards, sec_panel, config)
    stacked = pd.read_parquet(tmp_path / "derived" / "event_panel_stacked.parquet")
    stack_a = stacked[stacked["stack_id"] == "0000000001"]
    b_rows = stack_a[stack_a["cik"] == "0000000002"]
    assert (b_rows["period_end_date"] < pd.Timestamp("2020-12-31")).all()
