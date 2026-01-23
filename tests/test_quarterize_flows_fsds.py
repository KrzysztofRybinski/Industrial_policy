import pandas as pd

from industrial_policy.ingest.sec_fsds import compute_quarterly_from_fsds


def test_quarterize_prefers_direct_quarterly() -> None:
    df = pd.DataFrame(
        [
            {"cik": "0000000001", "tag": "REV", "fy": 2023, "period_end_date": "2023-03-31", "qtrs": 1, "value": 10.0},
            {"cik": "0000000001", "tag": "REV", "fy": 2023, "period_end_date": "2023-06-30", "qtrs": 1, "value": 12.0},
            {"cik": "0000000001", "tag": "REV", "fy": 2023, "period_end_date": "2023-06-30", "qtrs": 2, "value": 30.0},
            {"cik": "0000000001", "tag": "REV", "fy": 2023, "period_end_date": "2023-09-30", "qtrs": 3, "value": 70.0},
        ]
    )
    df["period_end_date"] = pd.to_datetime(df["period_end_date"])
    result = compute_quarterly_from_fsds(df, flow_tags={"REV"})

    q1 = result.loc[result["period_end_date"] == pd.Timestamp("2023-03-31")]
    assert q1.loc[q1["qtrs"] == 1, "quarterly_value"].iloc[0] == 10.0

    q2 = result.loc[result["period_end_date"] == pd.Timestamp("2023-06-30")]
    assert q2.loc[q2["qtrs"] == 1, "quarterly_value"].iloc[0] == 12.0
    assert q2.loc[q2["qtrs"] == 2, "quarterly_value"].isna().iloc[0]

    q3 = result.loc[result["period_end_date"] == pd.Timestamp("2023-09-30")]
    assert q3.loc[q3["qtrs"] == 3, "quarterly_value"].iloc[0] == 40.0
