"""Name matching for recipients to SEC companies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from rapidfuzz import process

from industrial_policy.log import get_logger
from industrial_policy.utils.sec import normalize_cik
from industrial_policy.utils.textnorm import normalize_name


def _match_names(
    names: pd.DataFrame,
    name_col: str,
    norm_col: str,
    choices: list[str],
    threshold: int,
    source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matches = []
    candidates = []
    for _, row in names.iterrows():
        name_norm = row.get(norm_col) or normalize_name(row.get(name_col, ""))
        if not name_norm:
            continue
        best = process.extract(name_norm, choices, limit=5)
        for rank, (match_name, score, _index) in enumerate(best, start=1):
            candidates.append(
                {
                    "name_raw": row.get(name_col),
                    "name_norm": name_norm,
                    "candidate_name": match_name,
                    "score": score,
                    "rank": rank,
                    "match_source": source,
                }
            )
        top_match_name, score, match_index = best[0]
        cik = None
        if score >= threshold:
            cik = match_index
        matches.append(
            {
                "name_raw": row.get(name_col),
                "name_norm": name_norm,
                "cik_index": cik,
                "score": score,
                "matched_name": top_match_name,
                "match_source": source,
            }
        )
    return pd.DataFrame(matches), pd.DataFrame(candidates)


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
    company_lookup["cik"] = company_lookup["cik"].map(normalize_cik)
    choices = company_lookup["company_name_norm"].tolist()

    awards = awards.copy()
    awards["recipient_name_norm"] = awards["recipient_name_norm"].fillna("").map(normalize_name)
    if "recipient_parent_name" in awards.columns:
        awards["recipient_parent_name_norm"] = awards["recipient_parent_name"].fillna("").map(
            normalize_name
        )

    unique_recipients = awards[["recipient_name", "recipient_name_norm"]].drop_duplicates()
    unique_parents = pd.DataFrame()
    if "recipient_parent_name_norm" in awards.columns:
        unique_parents = awards[
            ["recipient_parent_name", "recipient_parent_name_norm"]
        ].drop_duplicates()

    recipient_matches, recipient_candidates = _match_names(
        unique_recipients,
        "recipient_name",
        "recipient_name_norm",
        choices,
        threshold,
        "recipient",
    )
    parent_matches = pd.DataFrame()
    parent_candidates = pd.DataFrame()
    if not unique_parents.empty:
        parent_matches, parent_candidates = _match_names(
            unique_parents,
            "recipient_parent_name",
            "recipient_parent_name_norm",
            choices,
            threshold,
            "parent",
        )

    matches_df = recipient_matches.copy()
    candidates_df = pd.concat([recipient_candidates, parent_candidates], ignore_index=True)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(
            ["score", "matched_name", "name_norm"],
            ascending=[False, True, True],
        ).drop_duplicates(subset=["name_norm"], keep="first")
        matches_df["cik"] = matches_df["cik_index"].map(company_lookup["cik"])

    parent_matches_df = parent_matches.copy()
    if not parent_matches_df.empty:
        parent_matches_df = parent_matches_df.sort_values(
            ["score", "matched_name", "name_norm"],
            ascending=[False, True, True],
        ).drop_duplicates(subset=["name_norm"], keep="first")
        parent_matches_df["cik"] = parent_matches_df["cik_index"].map(company_lookup["cik"])

    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path)
        overrides["recipient_name_norm"] = overrides["recipient_name"].fillna("").map(normalize_name)
        overrides["cik"] = overrides["cik"].map(normalize_cik)
        matches_df = matches_df.merge(
            overrides[["recipient_name_norm", "cik"]],
            left_on="name_norm",
            right_on="recipient_name_norm",
            how="left",
            suffixes=("", "_override"),
        )
        matches_df["cik"] = matches_df["cik_override"].fillna(matches_df["cik"])
        matches_df = matches_df.drop(columns=["cik_override", "recipient_name_norm"])

    matches_path = data_dir / "recipient_cik_matches.parquet"
    recipient_matches_path = data_dir / "recipient_cik_matches_recipient.parquet"
    parent_matches_path = data_dir / "recipient_cik_matches_parent.parquet"
    candidates_path = data_dir / "recipient_cik_candidates_top5.parquet"
    matches_df.to_parquet(matches_path, index=False)
    recipient_matches.to_parquet(recipient_matches_path, index=False)
    parent_matches_df.to_parquet(parent_matches_path, index=False)
    candidates_df.to_parquet(candidates_path, index=False)

    awards_with_matches = awards.merge(
        matches_df.rename(
            columns={"name_norm": "recipient_name_norm", "score": "recipient_score"}
        ),
        on="recipient_name_norm",
        how="left",
    )
    if not parent_matches_df.empty and "recipient_parent_name_norm" in awards.columns:
        awards_with_matches = awards_with_matches.merge(
            parent_matches_df.rename(
                columns={"name_norm": "recipient_parent_name_norm", "score": "parent_score"}
            ),
            on="recipient_parent_name_norm",
            how="left",
            suffixes=("", "_parent"),
        )
        awards_with_matches["parent_score"] = awards_with_matches["parent_score"].fillna(-1)
        awards_with_matches["recipient_score"] = awards_with_matches["recipient_score"].fillna(-1)
        use_parent = (
            (awards_with_matches["parent_score"] > awards_with_matches["recipient_score"])
            & (awards_with_matches["parent_score"] >= threshold)
        )
        awards_with_matches["match_source"] = "recipient"
        awards_with_matches.loc[use_parent, "match_source"] = "parent"
        awards_with_matches.loc[use_parent, "cik"] = awards_with_matches.loc[use_parent, "cik_parent"]
        awards_with_matches = awards_with_matches.drop(columns=["cik_parent"])
    else:
        awards_with_matches["match_source"] = "recipient"

    if len(awards_with_matches) != len(awards):
        raise ValueError(
            "Recipient matching produced a different number of rows than awards; "
            "check for duplicate recipient_name_norm entries."
        )

    match_counts = awards_with_matches["match_source"].value_counts(dropna=False).to_dict()
    logger.info("Recipient matching sources: %s", match_counts)

    if "award_amount" in awards_with_matches.columns:
        unmatched = awards_with_matches[awards_with_matches["cik"].isna()]
        if not unmatched.empty:
            top_unmatched = unmatched.sort_values("award_amount", ascending=False).head(50)
            unmatched_path = Path(config["project"]["outputs_dir"]) / "tables" / "unmatched_large_awards.csv"
            unmatched_path.parent.mkdir(parents=True, exist_ok=True)
            top_unmatched.to_csv(unmatched_path, index=False)

    linked_path = data_dir / "awards_with_cik.parquet"
    awards_with_matches.to_parquet(linked_path, index=False)
    logger.info("Saved recipient matches to %s", matches_path)
    return awards_with_matches
