"""Vision backbone helper class."""

from typing import Any, Callable, Dict, List

import torch
from torch import nn
from typing_extensions import override

from eva.core.models import wrappers
from eva.vision.models.networks.backbones import BackboneModelRegistry


class VisionBackbone(wrappers.BaseModel):
    """Wrapper class for vision backbone models.

    This class can be used by load backbones available in eva's
    model registry by name. New backbones can be registered by using
    the `@register_model(model_name)` decorator.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Dict[str, Any] | None = None,
        tensor_transforms: Callable | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name: The name of the model to load.
            model_kwargs: The arguments used for instantiating the model.
            tensor_transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(tensor_transforms=tensor_transforms)

        self._model_name = model_name
        self._model_kwargs = model_kwargs

        self._model: nn.Module

        self.load_model()

    @override
    def load_model(self) -> None:
        self._model = BackboneModelRegistry.load_model(self._model_name, **self._model_kwargs or {})
        VisionBackbone.__name__ = self._model_name

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        return self._model(tensor)
