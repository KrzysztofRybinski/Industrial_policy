"""Configuration loader."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from industrial_policy.utils.paths import ensure_dirs


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load configuration from YAML.

    Args:
        path: Path to configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    logger = logging.getLogger(__name__)
    sec_user_agent = os.getenv("SEC_USER_AGENT")
    if sec_user_agent:
        config.setdefault("sec", {})["user_agent"] = sec_user_agent
        logger.info("SEC_USER_AGENT set; overriding config value")

    ensure_dirs(config)

    return config
