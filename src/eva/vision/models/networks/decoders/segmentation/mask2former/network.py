"""Mask2Former semantic segmentation decoder."""

from typing import List, Tuple

import torch
from torch import nn
from transformers.models.mask2former import modeling_mask2former

from eva.vision.models.networks.decoders.segmentation.mask2former import nn_layers


class Mask2FormerModel(nn.Module):
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
        """Initializes the Mask2Former model.

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
                nn_layers.Mask2FormerMaskedAttentionDecoderLayer(
                    self._embed_dim, self._num_attn_heads
                )
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

    def forward(
        self,
        patch_embeddings: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward of the decoder layers.

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
