"""Central logging configuration."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, honoring the ``EES_LOG_LEVEL`` env var."""

    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("EES_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""

    setup_logging()
    return logging.getLogger(name)
