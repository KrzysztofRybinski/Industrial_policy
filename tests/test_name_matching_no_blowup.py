from pathlib import Path

import pandas as pd

from industrial_policy.entity.name_matching import match_recipients
from industrial_policy.utils.textnorm import normalize_name


def test_name_matching_does_not_expand_rows(tmp_path: Path) -> None:
    awards = pd.DataFrame(
        {
            "recipient_name": ["Acme Corp", "Acme Corp", "Beta LLC"],
            "recipient_name_norm": [normalize_name("Acme Corp"), normalize_name("Acme Corp"), normalize_name("Beta LLC")],
        }
    )
    company_lookup = pd.DataFrame(
        {
            "company_name": ["Acme Corporation", "Beta LLC"],
            "company_name_norm": [normalize_name("Acme Corporation"), normalize_name("Beta LLC")],
            "cik": ["1234", "5678"],
        }
    )
    config = {
        "project": {"data_dir": str(tmp_path)},
        "matching": {
            "recipient_to_cik": {
                "fuzzy_threshold": 0,
                "manual_override_csv": str(tmp_path / "missing.csv"),
            }
        },
    }

    matched = match_recipients(awards, company_lookup, config)
    assert len(matched) == len(awards)
