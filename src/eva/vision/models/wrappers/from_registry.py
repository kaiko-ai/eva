"""Vision backbone helper class."""

from typing import Any, Callable, Dict

import torch
from typing_extensions import override

from eva.core.models.wrappers import base
from eva.core.utils import factory
from eva.vision.models.networks.backbones import backbone_registry


class ModelFromRegistry(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Wrapper class for vision backbone models.

    This class can be used by load backbones available in eva's
    model registry by name. New backbones can be registered by using
    the `@backbone_registry.register(model_name)` decorator.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Dict[str, Any] | None = None,
        model_extra_kwargs: Dict[str, Any] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name: The name of the model to load.
            model_kwargs: The arguments used for instantiating the model.
            model_extra_kwargs: Extra arguments used for instantiating the model.
            transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(transforms=transforms)

        self._model_name = model_name
        self._model_kwargs = model_kwargs or {}
        self._model_extra_kwargs = model_extra_kwargs or {}

        self.load_model()

    @override
    def load_model(self) -> None:
        self._model = factory.ModuleFactory(
            registry=backbone_registry,
            name=self._model_name,
            init_args=self._model_kwargs | self._model_extra_kwargs,
        )

        ModelFromRegistry.__name__ = self._model_name
