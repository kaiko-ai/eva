"""Convolutional based segmentation decoder."""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import override


class ConvDecoder(nn.Module):
    """Convolutional segmentation decoder."""

    def __init__(
        self,
        network: nn.Module,
        grid_size: int | Tuple[int, int],
    ) -> None:
        """Convolution based semantic segmentation decoder.

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

    def _preprocess(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Applies post-process transforms to the logits.

        We perform the following reshapes:
            (batch_size, number of patches, hidden_size)
             |
            (batch_size, hidden_size, number of patches)
             |
            (batch_size, hidden_size, num_patches_height, num_patches_width)

        Args:
            logits: The model outputs (batch_size, number of patches, num_classes).

        Returns:
            A tensor containing scores as predicted by the model for all of the
            classes (batch_size, num_classes, num_patches_height, num_patches_width).
        """
        patch_embeddings = patch_embeddings.permute(0, 2, 1)
        return patch_embeddings.reshape(patch_embeddings.shape[0], -1, *self._grid_size)

    def _model_forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Performs the model forward pass.

        It maps the embeddings of each patch to a segmentation class.

        Args:
            patch_embeddings: An embedding vector for each image patch
                (batch_size, number of patches, hidden_size).

        Returns:
            The logits as a tensor (batch_size, number of patches, num_classes).
        """
        return self._network(patch_embeddings)

    def _postprocess(
        self,
        logits: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Applies post-process transforms to the logits.

        Args:
            logits: The model outputs with shape:
                (batch_size, num_classes, downsampled_height, downsampled_width).
            image_size: The target image size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, num_classes, num_patches_height, num_patches_width).
        """
        return functional.interpolate(logits, image_size, mode="bilinear")

    @override
    def forward(self, patch_embeddings: torch.Tensor, image_size: Tuple[int, int]):
        embeddings = self._preprocess(patch_embeddings)
        logits = self._model_forward(embeddings)
        return self._postprocess(logits, image_size)
