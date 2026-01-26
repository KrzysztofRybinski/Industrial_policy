from __future__ import annotations

from datetime import date

import pandas as pd

from industrial_policy.features.panel import build_event_panel


def test_stacked_panel_includes_treated_rows(tmp_path):
    treated_cik = "0000000001"
    awards = pd.DataFrame(
        {
            "cik": [treated_cik],
            "treated": [1],
            "award_date": [date(2020, 1, 15)],
            "award_amount": [1_000_000],
        }
    )
    sec_panel = pd.DataFrame(
        {
            "cik": [treated_cik, treated_cik, "0000000002", "0000000002"],
            "period_end_date": [
                date(2019, 12, 31),
                date(2020, 3, 31),
                date(2019, 12, 31),
                date(2020, 3, 31),
            ],
        }
    )
    config = {
        "project": {"data_dir": str(tmp_path)},
        "panel": {
            "event_window_quarters": {"pre": 1, "post": 1},
            "treatment_definition": {"use_first_award_per_firm": True},
        },
    }

    build_event_panel(awards, sec_panel, config)
    stacked = pd.read_parquet(tmp_path / "derived" / "event_panel_stacked.parquet")
    assert (stacked["treated_in_stack"] == 1).any()

