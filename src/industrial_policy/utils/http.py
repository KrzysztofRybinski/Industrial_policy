"""HTTP helpers with caching and throttling."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from industrial_policy.log import get_logger


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_file(
    url: str,
    destination: Path,
    headers: Optional[Dict[str, str]] = None,
    sleep_seconds: float = 0.0,
) -> Path:
    """Download a file with basic caching and retries.

    Args:
        url: Source URL.
        destination: Output path.
        headers: Optional request headers.
        sleep_seconds: Throttle duration after download.

    Returns:
        Path to the downloaded file.
    """
    logger = get_logger()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("Cache hit for %s", destination)
        return destination

    logger.info("Downloading %s", url)
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return destination


def file_checksum(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
