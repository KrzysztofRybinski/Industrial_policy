"""Name matching for recipients to SEC companies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rapidfuzz import process

from industrial_policy.log import get_logger
from industrial_policy.utils.textnorm import normalize_name


def match_recipients(
    awards: pd.DataFrame,
    company_lookup: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Match award recipient names to SEC CIKs using fuzzy matching.

    Args:
        awards: USAspending awards dataframe.
        company_lookup: SEC company lookup dataframe.
        config: Loaded configuration.

    Returns:
        Awards dataframe with CIKs attached.
    """
    logger = get_logger()
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    data_dir.mkdir(parents=True, exist_ok=True)

    threshold = config["matching"]["recipient_to_cik"]["fuzzy_threshold"]
    overrides_path = Path(config["matching"]["recipient_to_cik"]["manual_override_csv"])

    company_lookup = company_lookup.copy()
    company_lookup["company_name_norm"] = company_lookup["company_name_norm"].fillna("")
    choices = company_lookup["company_name_norm"].tolist()

    matches = []
    candidates = []
    for _, row in awards.iterrows():
        name = row.get("recipient_name_norm") or normalize_name(row.get("recipient_name", ""))
        if not name:
            continue
        best = process.extract(name, choices, limit=5)
        for rank, (match_name, score, _index) in enumerate(best, start=1):
            candidates.append(
                {
                    "recipient_name": row.get("recipient_name"),
                    "recipient_name_norm": name,
                    "candidate_name": match_name,
                    "score": score,
                    "rank": rank,
                }
            )
        top_match_name, score, match_index = best[0]
        cik = company_lookup.iloc[match_index]["cik"] if score >= threshold else None
        matches.append(
            {
                "recipient_name": row.get("recipient_name"),
                "recipient_name_norm": name,
                "cik": cik,
                "score": score,
                "matched_name": top_match_name,
            }
        )

    matches_df = pd.DataFrame(matches)
    candidates_df = pd.DataFrame(candidates)

    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path)
        overrides["recipient_name_norm"] = overrides["recipient_name"].fillna("").map(normalize_name)
        matches_df = matches_df.merge(
            overrides[["recipient_name_norm", "cik"]],
            on="recipient_name_norm",
            how="left",
            suffixes=("", "_override"),
        )
        matches_df["cik"] = matches_df["cik_override"].fillna(matches_df["cik"])
        matches_df = matches_df.drop(columns=["cik_override"])

    matches_path = data_dir / "recipient_cik_matches.parquet"
    candidates_path = data_dir / "recipient_cik_candidates_top5.parquet"
    matches_df.to_parquet(matches_path, index=False)
    candidates_df.to_parquet(candidates_path, index=False)

    awards_with_matches = awards.merge(matches_df, on="recipient_name_norm", how="left")
    linked_path = data_dir / "awards_with_cik.parquet"
    awards_with_matches.to_parquet(linked_path, index=False)
    logger.info("Saved recipient matches to %s", matches_path)
    return awards_with_matches
