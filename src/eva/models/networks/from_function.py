"""Helper function from models defined with a function."""

from typing import Any, Callable, Dict

import jsonargparse
import torch
from loguru import logger
from torch import nn
from typing_extensions import override


class ModelFromFunction(nn.Module):
    """Wrapper class for models which are initialized from functions.

    This is helpful for initializing models in a `.yaml` configuration file.
    """

    def __init__(
        self,
        path: Callable[..., nn.Module],
        arguments: Dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            path: The path to the callable object (class or function).
            arguments: The extra callable function / class arguments.
            checkpoint_path: The path to the checkpoint to load the model weights from.

        Example:
            >>> import torchvision
            >>> network = ModelFromFunction(
            >>>     path=torchvision.models.resnet18,
            >>>     arguments={
            >>>         "weights": torchvision.models.ResNet18_Weights.DEFAULT,
            >>>     },
            >>> )
        """
        super().__init__()

        self._path = path
        self._arguments = arguments
        self._checkpoint_path = checkpoint_path

        self._network = self.build_model()

    def build_model(self) -> nn.Module:
        """Builds and returns the model."""
        class_path = jsonargparse.class_from_function(self._path, func_return=nn.Module)
        model = class_path(**self._arguments or {})
        if self._checkpoint_path is not None:
            model = self.load_model_checkpoint(model)
        return model

    def load_model_checkpoint(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Initializes the model with the weights.

        Args:
            model: model to initialize.

        Returns:
            the model initialized with the checkpoint.
        """
        logger.info(f"Loading {model.__class__.__name__} from checkpoint {self._checkpoint_path}")

        with open(self._checkpoint_path, "rb") as f:  # type: ignore
            checkpoint = torch.load(f, map_location="cpu")  # type: ignore[arg-type]
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=True)
            logger.info(
                f"Loaded modules for {model.__class__.__name__} from checkpoint "
                f"{self._checkpoint_path}"
            )
        return model

    @override
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._network(tensor)
