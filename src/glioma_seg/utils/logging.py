"""Lightweight logging helpers."""

import logging
import sys


def get_logger(name: str = "glioma_seg", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints to stdout.

    If the root logger is already configured (as it is inside a Hydra job,
    which also writes ``<run dir>/<script>.log``), messages propagate to it.
    Otherwise a stdout handler is attached. Idempotent: calling repeatedly
    will not stack handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers or logging.getLogger().handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
