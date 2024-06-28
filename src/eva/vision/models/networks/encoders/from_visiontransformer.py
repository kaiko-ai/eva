"""Encoder wrapper for timm models."""

from typing import Any, Callable, Dict, List

import jsonargparse
import numpy as np
import torch
from torch import nn
from typing_extensions import override

from eva.core.models.networks import _utils
from eva.vision.models.networks.encoders import encoder


class VisionTransformerEncoder(encoder.Encoder):
    """Encoder wrapper for torchvision vision_transformer models."""

    def __init__(
        self,
        path: Callable[..., nn.Module],
        arguments: Dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
        tensor_transforms: Callable | None = None,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            path: The path to the callable object (class or function).
            arguments: The extra callable function / class arguments.
            checkpoint_path: The path to the checkpoint to load the model
                weights from. This is currently only supported for torch
                model checkpoints. For other formats, the checkpoint loading
                should be handled within the provided callable object in <path>.
            tensor_transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        super().__init__()

        self._path = path
        self._arguments = arguments
        self._checkpoint_path = checkpoint_path
        self._tensor_transforms = tensor_transforms

        self._model = self.load_model()

    @override
    def load_model(self) -> nn.Module:
        class_path = jsonargparse.class_from_function(self._path, func_return=nn.Module)
        model = class_path(**self._arguments or {})
        if self._checkpoint_path is not None:
            _utils.load_model_weights(model, self._checkpoint_path)
        return model

    @override
    def forward(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        self._model.eval()
        features = []

        def hook(module, input, output):
            features.append(output)

        handle = self._model.blocks[-1].register_forward_hook(hook)
        with torch.no_grad():
            _ = self._model(tensor)
        handle.remove()
        # first embedding is the class token, remove it
        features = features[0][:, 1:, :]
        batch_size, n_patches, embedding_dim = features.shape
        patch_dim = int(np.sqrt(n_patches))
        features = features.view(batch_size, patch_dim, patch_dim, embedding_dim).permute(
            0, 3, 1, 2
        )
        return [features]
