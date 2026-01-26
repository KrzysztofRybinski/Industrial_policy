from __future__ import annotations

import pandas as pd

from industrial_policy.ingest.sec_fsds import build_sec_firm_period_panel


def test_sec_panel_unique_key():
    df = pd.DataFrame(
        {
            "cik": ["0000000001", "0000000001", "0000000001"],
            "period_end_date": [
                pd.Timestamp("2023-03-31"),
                pd.Timestamp("2023-03-31"),
                pd.Timestamp("2023-06-30"),
            ],
            "accepted_datetime": [
                pd.Timestamp("2023-05-01"),
                pd.Timestamp("2023-05-02"),
                pd.Timestamp("2023-08-01"),
            ],
            "fy": [2023, 2023, 2023],
            "fp": ["Q1", "Q1", "Q2"],
            "form": ["10-Q", "10-Q", "10-Q"],
            "sic": ["1234", "1234", "1234"],
            "tag": ["REV", "REV", "REV"],
            "metric_value": [100.0, 110.0, 120.0],
        }
    )
    panel = build_sec_firm_period_panel(df, {"revenue": ["REV"]})
    assert panel.duplicated(subset=["cik", "period_end_date"]).sum() == 0
