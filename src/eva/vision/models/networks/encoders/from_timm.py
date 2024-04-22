from typing import Any

import timm
import torch
from torch import nn


class TimmEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: str = "",
        keep_class_embedding: bool = False,
        **kwrgs: Any,
        # arguments: Dict[str, Any] | None = None,
    ) -> None:
        """Args:
        model_name: Name of model to instantiate.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        checkpoint_path: Path of checkpoint to load _after_ the model is initialized.
        kwrgs: Extra model arguments.
        """
        super().__init__()

        self._model_name = model_name
        self._pretrained = pretrained
        self._checkpoint_path = checkpoint_path
        self._keep_class_embedding = keep_class_embedding
        self._kwrgs = kwrgs

        self._network = self._load_model()

    def _load_model(self) -> nn.Module:
        """Builds and loads the model."""
        return timm.create_model(
            model_name=self._model_name,
            pretrained=self._pretrained,
            checkpoint_path=self._checkpoint_path,
            num_classes=0,
            **self._kwrgs,
        )

    def _remove_class_embedding(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        if patch_embeddings.dim() == 4:
            patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        else:
            patch_embeddings = patch_embeddings[:, self._network.num_prefix_tokens :]
        return patch_embeddings

    def forward(self, tensor: torch.Tensor):
        """Returns patch embeddings"""
        patch_embeddings = self._network.forward_features(tensor)
        return self._remove_class_embedding(patch_embeddings)
