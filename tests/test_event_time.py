from datetime import datetime

import pandas as pd

from industrial_policy.features.panel import build_event_panel


def test_event_time_indexing() -> None:
    awards = pd.DataFrame(
        {
            "cik": ["0001"],
            "treated": [1],
            "award_date": [pd.Timestamp("2020-06-15")],
        }
    )
    sec_panel = pd.DataFrame(
        {
            "cik": ["0001", "0001", "0001"],
            "period_end_date": [
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-09-30"),
            ],
        }
    )
    config = {
        "project": {"data_dir": "data"},
        "panel": {
            "event_window_quarters": {"pre": 1, "post": 1},
            "treatment_definition": {
                "use_first_award_per_firm": True,
                "min_award_amount": 0,
            },
        },
    }

    event_panel, _ = build_event_panel(awards, sec_panel, config)
    event_times = event_panel.sort_values("period_end_date")["event_time_q"].tolist()
    assert event_times == [-1, 0, 1]
