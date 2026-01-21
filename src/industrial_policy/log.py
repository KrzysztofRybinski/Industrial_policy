"""Logging helpers for the pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Path, name: str = "industrial_policy") -> logging.Logger:
    """Configure logging to console and a rotating file.

    Args:
        log_dir: Directory where log files are written.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger by name."""
    return logging.getLogger(name or "industrial_policy")
