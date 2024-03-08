"""Operations which are executed with the package import."""

import os
import sys
import warnings

import jsonargparse
from lightning_fabric.utilities import seed as pl_seed
from loguru import logger

from eva.utils import logo, workers


def _configure_random_seed(seed: int | None = None) -> None:
    """Sets the global random seed.

    Args:
        seed: The seed number to use. If `None`, it will read the seed from
            `EVA_GLOBAL_SEED` env variable. If `None` and the `EVA_GLOBAL_SEED`
            env variable is not set, then the seed defaults to `42`.
    """
    effective_seed = seed or os.environ.get("EVA_GLOBAL_SEED", default=42)
    pl_seed.seed_everything(seed=int(effective_seed), workers=True)


def _configure_jsonargparse() -> None:
    """Configures the `jsonargparse` library."""
    jsonargparse.set_config_read_mode(
        urls_enabled=True,
        fsspec_enabled=True,
    )


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


@workers.main_worker_only
def setup() -> None:
    """Sets up the environment before the module is imported."""
    logo.print_cli_logo()
    _configure_random_seed()
    _configure_jsonargparse()
    _initialize_logger()
    _suppress_warnings()
    _enable_mps_fallback()


setup()
