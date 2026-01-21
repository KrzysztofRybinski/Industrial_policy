import pandas as pd

from industrial_policy.utils.dates import quarterize_flow


def test_quarterize_flow_ytd_to_quarter() -> None:
    ytd = pd.Series([10.0, 30.0, 60.0])
    qtrs = pd.Series([1, 2, 3])
    quarterly = quarterize_flow(ytd, qtrs)
    assert quarterly.tolist() == [10.0, 20.0, 30.0]
