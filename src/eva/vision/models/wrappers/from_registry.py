"""Vision backbone helper class."""

from typing import Any, Callable, Dict, List

import torch
from torch import nn
from typing_extensions import override

from eva.core.models import wrappers
from eva.vision.models.networks.backbones import BackboneModelRegistry


class ModelFromRegistry(wrappers.BaseModel):
    """Wrapper class for vision backbone models.

    This class can be used by load backbones available in eva's
    model registry by name. New backbones can be registered by using
    the `@register_model(model_name)` decorator.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Dict[str, Any] | None = None,
        model_extra_kwargs: Dict[str, Any] | None = None,
        tensor_transforms: Callable | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name: The name of the model to load.
            model_kwargs: The arguments used for instantiating the model.
            model_extra_kwargs: Extra arguments used for instantiating the model.
            tensor_transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(tensor_transforms=tensor_transforms)

        self._model_name = model_name
        self._model_kwargs = model_kwargs or {}
        self._model_extra_kwargs = model_extra_kwargs or {}

        self._model: nn.Module

        self.load_model()

    @override
    def load_model(self) -> None:
        model_name = self._get_model_name()
        self._model = BackboneModelRegistry.load_model(
            self._model_name, **(self._model_kwargs | self._model_extra_kwargs)
        )
        ModelFromRegistry.__name__ = model_name

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        return self._model(tensor)

    def _get_model_name(self) -> str:
        if "MODEL_NAME" in (self._model_extra_kwargs or {}):
            return self._model_extra_kwargs.pop("MODEL_NAME")
        else:
            return self._model_name