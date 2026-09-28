"""Logging setup: one line per event to stdout, suitable for ``docker logs``."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once, and quiet noisy HTTP libraries.

    Args:
        level: Log level name for the root logger, e.g. ``"INFO"``.
    """
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, force=True)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
