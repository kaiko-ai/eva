"""Model wrapper for torch.hub models."""

from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from typing_extensions import override

from eva.core.models.wrappers import _utils, base


class TorchHubModel(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Model wrapper for `torch.hub` models."""

    def __init__(
        self,
        model_name: str,
        repo_or_dir: str,
        pretrained: bool = True,
        checkpoint_path: str = "",
        out_indices: int | Tuple[int, ...] | None = None,
        norm: bool = False,
        trust_repo: bool = True,
        model_kwargs: Dict[str, Any] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_name: Name of model to instantiate.
            repo_or_dir: The torch.hub repository or local directory to load the model from.
            pretrained: If set to `True`, load pretrained ImageNet-1k weights.
            checkpoint_path: Path of checkpoint to load.
            out_indices: Returns last n blocks if `int`, all if `None`, select
                matching indices if sequence.
            norm: Wether to apply norm layer to all intermediate features. Only
                used when `out_indices` is not `None`.
            trust_repo: If set to `False`, a prompt will ask the user whether the
                repo should be trusted.
            model_kwargs: Extra model arguments.
            transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__(transforms=transforms)

        self._model_name = model_name
        self._repo_or_dir = repo_or_dir
        self._pretrained = pretrained
        self._checkpoint_path = checkpoint_path
        self._out_indices = out_indices
        self._norm = norm
        self._trust_repo = trust_repo
        self._model_kwargs = model_kwargs or {}

        self.load_model()

    @override
    def load_model(self) -> None:
        """Builds and loads the torch.hub model."""
        self._model: nn.Module = torch.hub.load(
            repo_or_dir=self._repo_or_dir,
            model=self._model_name,
            trust_repo=self._trust_repo,
            pretrained=self._pretrained,
            **self._model_kwargs,
        )  # type: ignore

        if self._checkpoint_path:
            _utils.load_model_weights(self._model, self._checkpoint_path)

        TorchHubModel.__name__ = self._model_name

    @override
    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        if self._out_indices is not None:
            if not hasattr(self._model, "get_intermediate_layers"):
                raise ValueError(
                    "Only models with `get_intermediate_layers` are supported "
                    "when using `out_indices`."
                )

            return list(
                self._model.get_intermediate_layers(
                    tensor,
                    self._out_indices,
                    reshape=True,
                    return_class_token=False,
                    norm=self._norm,
                )
            )

        return self._model(tensor)
