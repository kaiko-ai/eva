"""Utilities and helper functions for models."""

from lightning_fabric.utilities import cloud_io
from loguru import logger
from torch import nn


def load_model_weights(model: nn.Module, checkpoint_path: str) -> None:
    """Loads (local or remote) weights to the model in-place.

    Args:
        model: The model to load the weights to.
        checkpoint_path: The path to the model weights/checkpoint.
    """
    logger.info(f"Loading '{model.__class__.__name__}' model from checkpoint '{checkpoint_path}'")

    fs = cloud_io.get_filesystem(checkpoint_path)
    with fs.open(checkpoint_path, "rb") as file:
        checkpoint = cloud_io._load(file, map_location="cpu")  # type: ignore
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        model.load_state_dict(checkpoint, strict=True)

    logger.info(f"Loading weights from '{checkpoint_path}' completed successfully.")
