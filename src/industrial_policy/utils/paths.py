"""Path utilities for ensuring pipeline directories exist."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def ensure_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Ensure standard data/output directories exist.

    Args:
        config: Loaded configuration.

    Returns:
        Dictionary of resolved paths.
    """
    project = config.setdefault("project", {})
    data_dir = Path(project.get("data_dir", "data"))
    outputs_dir = Path(project.get("outputs_dir", "outputs"))

    raw_dir = data_dir / "raw"
    cache_dir = data_dir / "cache"
    derived_dir = data_dir / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manual").mkdir(parents=True, exist_ok=True)

    (data_dir / "raw" / "sec_fsds").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw" / "usaspending").mkdir(parents=True, exist_ok=True)
    (data_dir / "derived" / "usaspending" / "chunks").mkdir(parents=True, exist_ok=True)

    outputs_tables = outputs_dir / "tables"
    outputs_figures = outputs_dir / "figures"
    outputs_logs = outputs_dir / "logs"
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)
    outputs_logs.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "outputs_dir": outputs_dir,
        "raw_dir": raw_dir,
        "cache_dir": cache_dir,
        "derived_dir": derived_dir,
        "outputs_tables": outputs_tables,
        "outputs_figures": outputs_figures,
        "outputs_logs": outputs_logs,
    }
