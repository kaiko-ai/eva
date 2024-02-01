"""Initializes the logger of the library.

This customizable logger can be used by just importing
`loguru` from everywhere as follows:
>>> from loguru import logger
>>> logger.info(...)
"""

import sys

from loguru import logger


def _initialize_logger() -> None:
    """Manipulates and customizes the logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<magenta>[{time:HH:mm:ss}]</magenta>"
        " <bold><level>{level}</level></bold> "
        " | {message}",
        colorize=True,
        level="INFO",
    )


_initialize_logger()
