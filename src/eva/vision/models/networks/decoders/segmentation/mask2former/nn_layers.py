"""Mask2Former building block."""

import torch
from torch import nn
from transformers.models.mask2former import modeling_mask2former


class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """A Transformer decoder block.

    It consisting of cross-attention, self-attention, and
    feed-forward layers, each followed by layer normalization.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_attn_heads: int = 8,
        decoder_ff_dim: int = 2048,
    ) -> None:
        """Initialize the decoder block.

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
            position_embeddings=(torch.zeros_like(q) if q_pos_embeds is None else q_pos_embeds),
        )
        q = q + residual
        q = self._self_attn_norm(q)

        # Feed-forward network forward pass
        residual = q
        q = self._ffn(q)
        q = q + residual
        q = self._final_layer_norm(q)

        return q
