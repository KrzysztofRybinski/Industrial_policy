from pathlib import Path

import pandas as pd

from industrial_policy.match.nearest_neighbor import nearest_neighbor_match


def test_nearest_neighbor_weights_sum_to_one(tmp_path: Path) -> None:
    propensity = pd.DataFrame(
        [
            {"cik": "0000000001", "event_year": 2020, "propensity_score": 0.6, "sic": 1000, "treated": 1},
            {"cik": "0000000002", "event_year": 2020, "propensity_score": 0.55, "sic": 1000, "treated": 0},
            {"cik": "0000000002", "event_year": 2019, "propensity_score": 0.52, "sic": 1000, "treated": 0},
            {"cik": "0000000003", "event_year": 2020, "propensity_score": 0.58, "sic": 1000, "treated": 0},
        ]
    )
    config = {"project": {"data_dir": str(tmp_path)}}

    matches = nearest_neighbor_match(propensity, config, k=2, caliper=0.2)
    weight_sum = matches.groupby("treated_cik")["weight"].sum().iloc[0]
    assert abs(weight_sum - 1.0) < 1e-9
