"""Mask2Former semantic segmentation decoder."""

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional
from transformers.models.mask2former import modeling_mask2former

from eva.vision.models.networks.decoders import decoder
from eva.vision.models.networks.decoders.segmentation.mask2former import decoder_block


class Mask2FormerDecoder(decoder.Decoder):
    """Mask2Former decoder for segmentation tasks using a transformer architecture."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        embed_dim: int = 256,
        num_queries: int = 100,
        num_attn_heads: int = 8,
        num_blocks: int = 9,
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

        self._in_features = in_features
        self._num_classes = num_classes
        self._embed_dim = embed_dim
        self._num_queries = num_queries
        self._num_attn_heads = num_attn_heads
        self._num_blocks = num_blocks

        self.projection_layer = nn.Sequential(
            nn.Conv2d(self._in_features, self._embed_dim, kernel_size=1),
            nn.GroupNorm(32, self._embed_dim),
        )
        self.q = nn.Embedding(self._num_queries, self._embed_dim)
        self.k_embed_pos = modeling_mask2former.Mask2FormerSinePositionEmbedding(
            num_pos_feats=self._embed_dim // 2, normalize=True
        )
        self.transformer_decoder = nn.ModuleList(
            [
                decoder_block.DecoderBlock(self._embed_dim, self._num_attn_heads)
                for _ in range(num_blocks)
            ]
        )
        self.q_pos_embed = nn.Embedding(self._num_queries, self._embed_dim)
        self.q_norm = nn.LayerNorm(self._embed_dim)
        self.q_mlp = modeling_mask2former.Mask2FormerMLPPredictionHead(
            input_dim=self._embed_dim,
            hidden_dim=self._embed_dim,
            output_dim=self._embed_dim,
        )
        self.q_class = nn.Linear(self._embed_dim, self._num_classes + 1)

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

    def _forward_decoder_layers(
        self, patch_embeddings: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward of the decoder head layers.

        Args:
            patch_embeddings: The patch embeddings tensor of shape
                (batch_size, hidden_size, n_patches_height, n_patches_width).

        Returns:
            The mask and class logits per layer. Specifically:
                mask_logits_per_layer: A list, of length equal to the number of
                    output classes, of the mask logits per layer, each of shape
                    (batch_size, num_queries, patch_height, patch_width).
                class_logits_per_layer: A list, of length equal to the number of
                    output classes, of the class logits per layer, each of shape
                    (batch_size, num_queries, num_classes).
        """
        x = self.projection_layer(patch_embeddings)

        q = self.q.weight
        q = q[:, None, :].repeat(1, x.shape[0], 1)
        v = x.view(self._embed_dim, x.shape[0], -1).transpose(0, 2)

        k = v + self.k_embed_pos(x).flatten(2).permute(2, 0, 1)

        q_pos_embeds = self.q_pos_embed.weight
        q_pos_embeds = q_pos_embeds[:, None, :].repeat(1, x.shape[0], 1)

        mask_logits_per_layer, class_logits_per_layer = [], []
        for block in self.transformer_decoder:
            attn_mask, mask_logits, class_logits = self._compute_logits_and_attention_mask(q, x)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)
            q = block(q, k, v, q_pos_embeds, attn_mask)

        _, mask_logits, class_logits = self._compute_logits_and_attention_mask(q, x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits_per_layer, class_logits_per_layer

    def _compute_logits_and_attention_mask(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper building block.

        Args:
            q: Query tensor.
            x: Input tensor.

        Returns:
            Attention mask, mask logits, and class logits.
        """
        q_intermediate = self.q_norm(q)
        class_logits = self.q_class(q_intermediate).transpose(0, 1)
        mask_logits = torch.einsum("qbc, bchw -> bqhw", self.q_mlp(q_intermediate), x)
        attn_mask = (mask_logits < 0).bool().flatten(-2)
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        return attn_mask, mask_logits, class_logits

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
        semantic_one_hot_mask = torch.zeros(
            mask_logits_per_layer[0].size(0),
            self._num_classes,
            *output_size,
            device=mask_logits_per_layer[0].device,
        )
        for mask_logits, class_logits in zip(
            mask_logits_per_layer, class_logits_per_layer, strict=True
        ):
            pixel_logits_semantic = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                class_logits.softmax(dim=-1)[..., :-1],  # drop the dummy class
            )
            semantic_one_hot_mask += functional.interpolate(
                pixel_logits_semantic, output_size, mode="bilinear", align_corners=False
            )
        return semantic_one_hot_mask

    def forward(
        self,
        features: List[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass for the Mask2formerDecoder.

        Args:
            features: A list of multi-level patch embeddings of shape
                (batch_size, hidden_size, path_height, path_width).
            output_size: The mask output size (height, width).

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        patch_embeddings = self._forward_features(features)
        mask_logits_per_layer, class_logits_per_layer = self._forward_decoder_layers(
            patch_embeddings
        )
        return self._cls_seg(
            mask_logits_per_layer,
            class_logits_per_layer,
            output_size=output_size,
        )
