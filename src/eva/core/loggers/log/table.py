"""Table log functionality."""

import functools
from typing import List

import pandas as pd

from eva.core.loggers import loggers
from eva.core.loggers.log import utils


@functools.singledispatch
def log_table(
    logger,
    tag: str,
    columns: List[str] | None = None,
    data: List[List[str]] | None = None,
    dataframe: pd.DataFrame | None = None,
    step: int = 0,
) -> None:
    """Adds a a table to the logger.

    The table can be defined either with `columns` and `data` or with `dataframe`.

    Args:
        logger: The logger to log the table to.
        tag: The log tag.
        columns: The column names of the table.
        data: The data of the table as a list of lists.
        dataframe: A pandas DataFrame to log.
        step: The global step of the log.
    """
    utils.raise_not_supported(logger, "table")


@log_table.register
def _(
    loggers: list,
    tag: str,
    columns: List[str] | None = None,
    data: List[List[str]] | None = None,
    dataframe: pd.DataFrame | None = None,
    step: int = 0,
) -> None:
    """Adds a table to a list of supported loggers."""
    for logger in loggers:
        log_table(
            logger,
            tag=tag,
            columns=columns,
            data=data,
            dataframe=dataframe,
            step=step,
        )


@log_table.register
def _(
    logger: loggers.WandbLogger,
    tag: str,
    columns: List[str] | None = None,
    data: List[List[str]] | None = None,
    dataframe: pd.DataFrame | None = None,
    step: int = 0,
) -> None:
    """Adds a table to a Wandb logger."""
    logger.log_table(
        key=tag,
        columns=columns,
        data=data,
        dataframe=dataframe,
        step=step,
    )
