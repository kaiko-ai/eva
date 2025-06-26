"""Model wrapper for timm models."""

from typing import Any, Callable, Dict, Tuple
from urllib import parse

import timm
import torch
from typing_extensions import override

from eva.core.models.wrappers import base


class TimmModel(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Model wrapper for `timm` models.

    Note that only models with `forward_intermediates`
    method are currently supported.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        checkpoint_path: str = "",
        out_indices: int | Tuple[int, ...] | None = None,
        model_kwargs: Dict[str, Any] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_name: Name of model to instantiate.
            pretrained: If set to `True`, load pretrained ImageNet-1k weights.
            checkpoint_path: Path of checkpoint to load.
            out_indices: Returns last n blocks if `int`, all if `None`, select
                matching indices if sequence.
            model_kwargs: Extra model arguments.
            transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(transforms=transforms)

        self._model_name = model_name
        self._pretrained = pretrained
        self._checkpoint_path = checkpoint_path
        self._out_indices = out_indices
        self._model_kwargs = model_kwargs or {}

        self.load_model()

    @override
    def load_model(self) -> None:
        """Builds and loads the timm model as feature extractor."""
        self._model = timm.create_model(
            model_name=self._model_name,
            pretrained=True if self._checkpoint_path else self._pretrained,
            pretrained_cfg=self._pretrained_cfg,
            out_indices=self._out_indices,
            features_only=self._out_indices is not None,
            **self._model_kwargs,
        )
        TimmModel.__name__ = self._model_name

    @property
    def _pretrained_cfg(self) -> Dict[str, Any]:
        if not self._checkpoint_path:
            return {}
        key = "file" if parse.urlparse(self._checkpoint_path).scheme in ("file", "") else "url"
        return {key: self._checkpoint_path, "num_classes": 0}
