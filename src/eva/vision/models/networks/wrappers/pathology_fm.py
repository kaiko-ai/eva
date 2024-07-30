"""Helper wrapper class for Pathology FMs."""

from typing import Any, Callable, Dict

import torch
from torch import nn
from typing_extensions import override

from eva.core.models.networks.wrappers import base
from eva.vision.models.networks.backbones.pathology import PathologyModelRegistry


class PathologyFM(base.BaseModel):
    """Wrapper class for Pathology FMs."""

    def __init__(
        self,
        model_name: str,
        arguments: Dict[str, Any] | None = None,
        tensor_transforms: Callable | None = None,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            model_name: The name of the model to load.
            arguments: The arguments used for instantiating the model.
            tensor_transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(tensor_transforms=tensor_transforms)

        self._model_name = model_name
        self._arguments = arguments
        self._model = self.load_model()

    @override
    def load_model(self) -> nn.Module:
        return PathologyModelRegistry.load_model(self._model_name, **self._arguments or {})

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._model(tensor)
