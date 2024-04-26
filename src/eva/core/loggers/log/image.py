"""Image log functionality."""

import functools

import torch

from eva.core.loggers import loggers
from eva.core.loggers.log import utils


@functools.singledispatch
def log_image(
    logger,
    tag: str,
    image: torch.Tensor,
    caption: str | None = None,
    step: int = 0,
) -> None:
    """Adds an image to the logger.

    Args:
        logger: The desired logger.
        tag: The log tag.
        image: The image tensor to log.
        caption: The image caption.
        step: The global step of the log. Defaults to `0`.
    """
    utils.raise_not_supported(logger, "image")


@log_image.register
def _(
    loggers: list,
    tag: str,
    image: torch.Tensor,
    step: int = 0,
) -> None:
    """Adds an image to a list of supported loggers."""
    for logger in loggers:
        log_image(
            logger,
            tag=tag,
            image=image,
            step=step,
        )


@log_image.register
def _(
    logger: loggers.TensorBoardLogger,
    tag: str,
    image: torch.Tensor,
    step: int = 0,
) -> None:
    """Adds an image to a TensorBoard logger."""
    logger.experiment.add_image(
        tag=tag,
        img_tensor=image,
        global_step=step,
    )
