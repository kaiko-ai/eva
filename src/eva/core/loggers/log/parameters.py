"""Text log functionality."""

import functools
from typing import Any, Dict

import yaml

from eva.core.loggers import experimental_loggers as loggers_lib
from eva.core.loggers.log import utils


@functools.singledispatch
def log_parameters(
    logger,
    tag: str,
    parameters: Dict[str, Any],
) -> None:
    """Adds parameters to the logger.

    Args:
        logger: The desired logger.
        tag: The log tag.
        parameters: The parameters to log.
    """
    utils.raise_not_supported(logger, "parameters")


@log_parameters.register
def _(
    loggers: list,
    tag: str,
    parameters: Dict[str, Any],
) -> None:
    """Adds parameters to a list of supported loggers."""
    for logger in loggers:
        log_parameters(logger, tag=tag, parameters=parameters)


@log_parameters.register
def _(
    logger: loggers_lib.TensorBoardLogger,
    tag: str,
    parameters: Dict[str, Any],
) -> None:
    """Adds parameters to a TensorBoard logger."""
    as_markdown_text = _yaml_to_markdown(parameters)
    logger.experiment.add_text(
        tag=tag,
        text_string=as_markdown_text,
        global_step=0,
    )


@log_parameters.register
def _(
    logger: loggers_lib.WandbLogger,
    tag: str,
    parameters: Dict[str, Any],
) -> None:
    """Adds parameters to a Wandb logger."""
    logger.experiment.config.update(parameters)


def _yaml_to_markdown(data: Dict[str, Any]) -> str:
    """Casts yaml data to markdown.

    Args:
        data: The yaml data.

    Returns:
        A string markdown friendly formatted.
    """
    text = yaml.dump(data, sort_keys=False)
    return f"```yaml\n{text}```"
