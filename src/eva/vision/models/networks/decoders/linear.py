"""Linear based semantic segmentation decoder."""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import override


class LinearDecoder(nn.Module):
    """Linear segmentation decoder."""

    def __init__(
        self,
        network: nn.Module,
        grid_size: int | Tuple[int, int],
    ) -> None:
        """Linear based semantic segmentation decoder.

        Args:
            network: The linear decoder network.
            grid_size: The number of vision patches (tokens) along the height
                and width dimensions of the input image. For example, if you
                have an input image of size 224x224 and use a patch size of
                16x16, then the grid size would be 14x14, because 224 divided
                by 16 equals 14.
        """
        super().__init__()

        self._network = network
        self._grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)

    def _model_forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Performs the model forward pass.

        It maps the embeddings of each patch to a segmentation class.

        Args:
            patch_embeddings: An embedding vector for each image patch
                (batch_size, number of patches, hidden_size).

        Returns:
            The tensor logits (batch_size, number of patches, num_classes).
        """
        return self._network(patch_embeddings)

    def _postprocess(
        self,
        logits: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Applies post-process transforms to the logits.

        We perform the following reshapes:
            (batch_size, num_classes, number of patches)
             |
            (batch_size, num_classes, num_patches_height, num_patches_width)
             |
            (batch_size, num_classes, image_height, image_width)

        Args:
            logits: The model outputs with shape:
                (batch_size, number of patches, num_classes).
            image_size: The target image size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, num_classes, num_patches_height, num_patches_width).
        """
        logits = logits.transpose(1, 2)
        logits = logits.reshape(logits.shape[0], -1, *self._grid_size)
        return functional.interpolate(logits, image_size, mode="bilinear")

    @override
    def forward(self, patch_embeddings: torch.Tensor, image_size: Tuple[int, int]):
        logits = self._model_forward(patch_embeddings)
        return self._postprocess(logits, image_size=image_size)
