"""Operations which are executed with the package import."""

import os
import sys
import warnings

from loguru import logger


def _initialize_logger() -> None:
    """Initializes, manipulates and customizes the logger.

    This customizable logger can be used by just importing `loguru`
    from everywhere as follows:
    >>> from loguru import logger
    >>> logger.info(...)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<magenta>[{time:HH:mm:ss}]</magenta>"
        " <bold><level>{level}</level></bold> "
        " | {message}",
        colorize=True,
        level="INFO",
    )


def _suppress_warnings() -> None:
    """Suppress all warnings from all subprocesses."""
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"


def _enable_mps_fallback() -> None:
    """It enables the MPS fallback in torch.

    Note that this action has to take place before importing torch.
    """
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup() -> None:
    """Sets up the environment before the module is imported."""
    _initialize_logger()
    _suppress_warnings()
    _enable_mps_fallback()


setup()
