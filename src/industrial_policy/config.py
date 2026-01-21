"""Configuration loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


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

    sec_user_agent = os.getenv("SEC_USER_AGENT")
    if sec_user_agent:
        config.setdefault("sec", {})["user_agent"] = sec_user_agent

    project = config.setdefault("project", {})
    data_dir = Path(project.get("data_dir", "data"))
    outputs_dir = Path(project.get("outputs_dir", "outputs"))
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "logs").mkdir(parents=True, exist_ok=True)
    (data_dir / "cache").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "derived").mkdir(parents=True, exist_ok=True)

    return config
