from __future__ import annotations

import pandas as pd

from industrial_policy.ingest.sec_fsds import compute_quarterly_from_fsds


def test_quarterize_prefers_qtrs_one():
    df = pd.DataFrame(
        {
            "cik": ["0001", "0001"],
            "tag": ["REV", "REV"],
            "fy": [2023, 2023],
            "qtrs": [1, 2],
            "period_end_date": [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-03-31")],
            "value": [100.0, 250.0],
        }
    )
    out = compute_quarterly_from_fsds(df, flow_tags={"REV"})
    q1_val = out.loc[out["qtrs"] == 1, "metric_value"].iloc[0]
    q2_val = out.loc[out["qtrs"] == 2, "metric_value"].iloc[0]
    assert q1_val == 100.0
    assert pd.isna(q2_val)
