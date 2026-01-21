"""Text normalization utilities."""
from __future__ import annotations

import re

SUFFIXES = [
    " INC",
    " INCORPORATED",
    " CORP",
    " CORPORATION",
    " LLC",
    " LTD",
    " LIMITED",
    " CO",
    " COMPANY",
]


def normalize_name(name: str) -> str:
    """Normalize company names for matching.

    Args:
        name: Raw name.

    Returns:
        Normalized name.
    """
    if not name:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", name.upper()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for suffix in SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
