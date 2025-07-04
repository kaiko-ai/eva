"""Wrappers for HuggingFace `transformers` models."""

from typing import Any, Callable, Dict

import torch
import transformers
from typing_extensions import override

from eva.core.models.wrappers import base


class HuggingFaceModel(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Wrapper class for loading HuggingFace `transformers` models."""

    def __init__(
        self,
        model_name_or_path: str,
        transforms: Callable | None = None,
        model_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            model_name_or_path: The model name or path to load the model from.
                This can be a local path or a model name from the `HuggingFace`
                model hub.
            transforms: The transforms to apply to the output tensor
                produced by the model.
            model_kwargs: The arguments used for instantiating the model.
        """
        super().__init__(transforms=transforms)

        self._model_name_or_path = model_name_or_path
        self._model_kwargs = model_kwargs or {}

        self.load_model()

    @override
    def load_model(self) -> None:
        # Use safetensors to avoid torch.load security vulnerability
        model_kwargs = {"use_safetensors": True, **self._model_kwargs}
        self._model = transformers.AutoModel.from_pretrained(
            self._model_name_or_path, **model_kwargs
        )
