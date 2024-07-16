"""Phikon model encoder."""

from typing import List

import torch
from typing_extensions import override

from eva.vision.models.networks.encoders import encoder
from eva.vision.models.networks import phikon


class PhikonEncoder(encoder.Encoder):
    """Encoder wrapper for Phikon."""

    def __init__(self) -> None:
        super().__init__()

        self._model = phikon.Phikon()

    @override
    def forward(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        features = self._model(tensor)
        patch_embeddings = features.view(features.shape[0], -1, 16, 16)
        return [patch_embeddings]
