"""Lightweight logging helpers."""

import logging
import sys


def get_logger(name: str = "glioma_seg", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger.

    Idempotent — calling repeatedly will not stack handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
