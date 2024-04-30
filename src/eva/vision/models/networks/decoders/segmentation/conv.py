"""Convolutional based semantic segmentation decoder."""

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional

from eva.vision.models.networks.decoders import decoder


class ConvDecoder(decoder.Decoder):
    """Convolutional segmentation decoder."""

    def __init__(self, layers: nn.Module) -> None:
        """Initializes the convolutional based decoder head.

        Here the input nn layers will be directly applied to the
        features of shape (batch_size, hidden_size, n_patches_height,
        n_patches_width), where n_patches is image_size / patch_size.
        Note the n_patches is also known as grid_size.

        Args:
            layers: The convolutional layers to be used as the decoder head.
        """
        super().__init__()

        self._layers = layers

    def _forward_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for multi-level feature maps to a single one.

        Args:
            features: List of multi-level image features of shape (batch_size,
                hidden_size, n_patches_height, n_patches_width).

        Returns:
            A tensor of shape (batch_size, hidden_size, n_patches_height,
            n_patches_width) which is feature map of the decoder head.
        """
        if not isinstance(features, list) and features[0].ndim != 4:
            raise ValueError(
                "Input features should be a list of four (4) dimensional inputs of "
                "shape (batch_size, hidden_size, n_patches_height, n_patches_width)."
            )

        return features[-1]

    def _forward_head(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward of the decoder head.

        Args:
            patch_embeddings: The patch embeddings tensor of shape
                (batch_size, hidden_size, n_patches_height, n_patches_width).

        Returns:
            The logits as a tensor (batch_size, n_classes, upscale_height, upscale_width).
        """
        return self._layers(patch_embeddings)

    def _cls_seg(
        self,
        logits: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Classify each pixel of the image.

        Args:
            logits: The decoder outputs of shape (batch_size, n_classes,
                height, width).
            image_size: The target image size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        return functional.interpolate(logits, image_size, mode="bilinear")

    def forward(
        self,
        features: List[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Maps the patch embeddings to a segmentation mask of the image size.

        Args:
            features: List of multi-level image features of shape (batch_size,
                hidden_size, n_patches_height, n_patches_width).
            image_size: The target image size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        patch_embeddings = self._forward_features(features)
        logits = self._forward_head(patch_embeddings)
        return self._cls_seg(logits, image_size)
