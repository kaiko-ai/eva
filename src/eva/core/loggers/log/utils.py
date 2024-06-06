"""Logging related utilities."""

from loguru import logger as cli_logger

from eva.core.loggers import ExperimentalLoggers


def raise_not_supported(logger: ExperimentalLoggers, data_type: str) -> None:
    """Raises a warning for not supported tasks from the given logger."""
    print("\n")
    cli_logger.debug(
        f"Logger '{logger.__class__.__name__}' is not supported for " f"'{data_type}' data."
    )
