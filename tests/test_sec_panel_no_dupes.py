import pandas as pd

from industrial_policy.ingest.sec_fsds import build_sec_firm_period_panel


def test_sec_panel_dedupes_firm_periods() -> None:
    df = pd.DataFrame(
        [
            {
                "cik": "0000000001",
                "period_end_date": pd.Timestamp("2023-03-31"),
                "accepted_datetime": pd.Timestamp("2023-04-01"),
                "fy": 2023,
                "fp": "Q1",
                "form": "10-Q",
                "sic": 1000,
                "tag": "REV",
                "metric_value": 10.0,
            },
            {
                "cik": "0000000001",
                "period_end_date": pd.Timestamp("2023-03-31"),
                "accepted_datetime": pd.Timestamp("2023-04-05"),
                "fy": 2023,
                "fp": "Q1",
                "form": "10-Q",
                "sic": 1000,
                "tag": "REV",
                "metric_value": 12.0,
            },
        ]
    )
    panel = build_sec_firm_period_panel(df, {"revenue": ["REV"]})
    assert not panel.duplicated(subset=["cik", "period_end_date"]).any()
