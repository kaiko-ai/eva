"""Mask2Former semantic segmentation decoder."""

from typing import List, Tuple

import torch
from torch.nn import functional

from eva.vision.models.networks.decoders import decoder
from eva.vision.models.networks.decoders.segmentation.mask2former import network


class Mask2FormerDecoder(decoder.Decoder):
    """Mask2Former decoder for segmentation tasks using a transformer architecture."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        embed_dim: int = 256,
        num_queries: int = 16,
        num_attn_heads: int = 1,
        num_blocks: int = 1,
        # embed_dim: int = 256,
        # num_queries: int = 100,
        # num_attn_heads: int = 8,
        # num_blocks: int = 9,
    ) -> None:
        """Initializes the decoder.

        Args:
            in_features: The hidden dimension size of the embeddings.
            num_classes: Number of output classes.
            num_queries: Number of query embeddings.
            num_attn_heads: Number of attention heads.
            num_blocks: Number of decoder blocks.
            embed_dim: Dimension of the embedding.
        """
        super().__init__()

        self._mask2former = network.Mask2FormerModel(
            in_features=in_features,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_attn_heads=num_attn_heads,
            num_blocks=num_blocks,
        )

    def _forward_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for multi-level feature maps to a single one.

        It will interpolate the features and concat them into a single tensor
        on the dimension axis of the hidden size.

        Example:
            >>> features = [torch.Tensor(16, 384, 14, 14), torch.Size(16, 384, 14, 14)]
            >>> output = self._forward_features(features)
            >>> assert output.shape == torch.Size([16, 768, 14, 14])

        Args:
            features: List of multi-level image features of shape (batch_size,
                hidden_size, n_patches_height, n_patches_width).

        Returns:
            A tensor of shape (batch_size, hidden_size, n_patches_height,
            n_patches_width) which is feature map of the decoder head.
        """
        if not isinstance(features, list) or features[0].ndim != 4:
            raise ValueError(
                "Input features should be a list of four (4) dimensional inputs of "
                "shape (batch_size, hidden_size, n_patches_height, n_patches_width)."
            )

        upsampled_features = [
            functional.interpolate(
                input=embeddings,
                size=features[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for embeddings in features
        ]
        return torch.cat(upsampled_features, dim=1)

    def _cls_seg(
        self,
        mask_logits_per_layer: List[torch.Tensor],
        class_logits_per_layer: List[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Classify each pixel of the input image.

        Args:
            mask_logits_per_layer: A list, of length equal to the number of
                output classes, of the mask logits per layer, each of shape
                (batch_size, num_queries, patch_height, patch_width).
            class_logits_per_layer: A list, of length equal to the number of
                output classes, of the class logits per layer, each of shape
                (batch_size, num_queries, num_classes).
            output_size: The target mask size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, mask_height, mask_width).
        """
        class_queries_logits = class_logits_per_layer[-1]
        masks_queries_logits = functional.interpolate(
            mask_logits_per_layer[-1], output_size, mode="bilinear", align_corners=False
        )
        seg_logits = torch.einsum(
            "bqhw, bqc -> bchw",
            masks_queries_logits.sigmoid(),
            class_queries_logits.softmax(dim=-1)[..., :-1],  # drop the dummy class
        )
        return seg_logits

    def forward(
        self,
        features: List[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Forward pass for the Mask2formerDecoder.

        Args:
            features: A list of multi-level patch embeddings of shape
                (batch_size, hidden_size, path_height, path_width).
            output_size: The mask output size (height, width).

        Returns:
            A tensor containing scores for all of the classes with
            shape (batch_size, n_classes, image_height, image_width).
        """
        patch_embeddings = self._forward_features(features)
        mask_logits_per_layer, class_logits_per_layer = self._mask2former(patch_embeddings)
        segmentation_logits = self._cls_seg(
            mask_logits_per_layer,
            class_logits_per_layer,
            output_size=output_size,
        )
        return segmentation_logits, (mask_logits_per_layer, class_logits_per_layer)
