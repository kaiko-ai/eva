"""Mask2Former semantic segmentation decoder."""

from typing import Tuple

import torch
from torch import nn
from transformers.models.mask2former import modeling_mask2former

from eva.vision.models.networks.decoders import decoder


class Mask2formerDecoder(decoder.Decoder):
    """Mask2Former decoder for segmentation tasks using a transformer architecture."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_queries: int = 100,
        num_attn_heads: int = 8,
        num_blocks: int = 9,
        embed_dim: int = 256,
    ) -> None:
        """Initializes the decoder.

        Args:
            in_features: Number of input features.
            num_classes: Number of output classes.
            num_queries: Number of query embeddings.
            num_attn_heads: Number of attention heads.
            num_blocks: Number of decoder blocks.
            embed_dim: Dimension of the embedding.
        """
        super().__init__()

        self._num_classes = num_classes
        self._num_queries = num_queries
        self._num_attn_heads = num_attn_heads
        self._num_blocks = num_blocks
        self._embed_dim = embed_dim

        self.projection_layer = nn.Linear(in_features, self._embed_dim)
        self.q = nn.Embedding(self._num_queries, self._embed_dim)
        self.k_embed_pos = modeling_mask2former.Mask2FormerSinePositionEmbedding(
            num_pos_feats=self._embed_dim // 2, normalize=True
        )
        self.transformer_decoder = nn.ModuleList(
            [DecoderBlock(self._embed_dim, self._num_attn_heads) for _ in range(num_blocks)]
        )
        self.q_pos_embed = nn.Embedding(self._num_queries, self._embed_dim)
        self.q_norm = nn.LayerNorm(self._embed_dim)
        self.q_mlp = modeling_mask2former.Mask2FormerMLPPredictionHead(
            input_dim=self._embed_dim,
            hidden_dim=self._embed_dim,
            output_dim=self._embed_dim,
        )
        self.q_class = nn.Linear(self._embed_dim, self._num_classes + 1)

    def _predict(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function to perform prediction.

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

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Mask2formerDecoder.

        Args:
            tensor: Input tensor of shape (batch_size, in_features, height, width).

        Returns:
            Mask logits and class logits per layer.
        """
        x = self.projection_layer(tensor)

        q = self.q.weight
        q = q[:, None, :].repeat(1, x.shape[0], 1)

        v = x.transpose(0, 1)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)
        k = v + self.k_embed_pos(x).flatten(2).permute(2, 0, 1)

        q_pos_embeds = self.q_pos_embed.weight
        q_pos_embeds = q_pos_embeds[:, None, :].repeat(1, x.shape[0], 1)

        mask_logits_per_layer, class_logits_per_layer = [], []
        for block in self.transformer_decoder:
            attn_mask, mask_logits, class_logits = self._predict(q, x)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)
            q = block(q, k, v, q_pos_embeds, attn_mask)

        _, mask_logits, class_logits = self._predict(q, x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits_per_layer, class_logits_per_layer


class DecoderBlock(nn.Module):
    """A Transformer decoder block.

    It consisting of cross-attention, self-attention, and
    feed-forward layers, each followed by layer normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        decoder_ff_dim: int = 2048,
    ) -> None:
        """Initialize the decoder black.

        Args:
            embed_dim: The dimension of the input embedding.
            num_attn_heads: The number of attention heads.
            decoder_ff_dim: The dimension of the feed-forward network.
        """
        super().__init__()

        self._embed_dim = embed_dim
        self._num_attn_heads = num_attn_heads
        self._decoder_ff_dim = decoder_ff_dim

        self._cross_attn = nn.MultiheadAttention(self._embed_dim, self._num_attn_heads)
        self._cross_attn_norm = nn.LayerNorm(self._embed_dim)

        self._self_attn = modeling_mask2former.Mask2FormerAttention(
            embed_dim=self._embed_dim, num_heads=self._num_attn_heads
        )
        self._self_attn_norm = nn.LayerNorm(self._embed_dim)

        self._ffn = nn.Sequential(
            nn.Linear(self._embed_dim, self._decoder_ff_dim),
            nn.ReLU(),
            nn.Linear(self._decoder_ff_dim, self._embed_dim),
        )
        self._final_layer_norm = nn.LayerNorm(self._embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_pos_embeds: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for the decoder block.

        Args:
            q: Query tensor of shape (seq_len, batch_size, embed_dim).
            k: Key tensor of shape (seq_len, batch_size, embed_dim).
            v: Value tensor of shape (seq_len, batch_size, embed_dim).
            q_pos_embeds: Positional embeddings for the query.
            mask: Attention mask.

        Returns:
            The output tensor of shape (seq_len, batch_size, embed_dim).
        """
        # Adjust mask for multi-head attention
        if mask is not None:
            mask = mask[:, None, ...].repeat(1, self._cross_attn.num_heads, 1, 1)
            mask = mask.flatten(0, 1)

        # Cross-attention forward pass
        residual = q
        q, _ = self._cross_attn(
            query=q if q_pos_embeds is None else q + q_pos_embeds,
            key=k,
            value=v,
            attn_mask=mask,
        )
        q = q + residual
        q = self._cross_attn_norm(q)

        # Self-attention forward pass
        residual = q
        q, _ = self._self_attn(
            hidden_states=q,
            position_embeddings=(q_pos_embeds or torch.zeros_like(q)),
        )
        q = q + residual
        q = self._self_attn_norm(q)

        # Feed-forward network forward pass
        residual = q
        q = self._ffn(q)
        q = q + residual
        q = self._final_layer_norm(q)

        return q
