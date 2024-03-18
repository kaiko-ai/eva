"""Wrappers for HuggingFace `transformers` models."""

from typing import Any, Callable

import torch
import transformers
from typing_extensions import override

from eva.core.models.networks.wrappers import base


class HuggingFaceModel(base.BaseModel):
    """Wrapper class for loading HuggingFace `transformers` models."""

    def __init__(self, model_name_or_path: str, tensor_transforms: Callable | None = None) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            tensor_transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(tensor_transforms=tensor_transforms)

        self._model_name_or_path = model_name_or_path
        self._model = self.load_model()

    @override
    def load_model(self) -> Any:
        config = transformers.AutoConfig.from_pretrained(self._model_name_or_path)
        return transformers.AutoModel.from_pretrained(self._model_name_or_path, config=config)

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._model(tensor)
