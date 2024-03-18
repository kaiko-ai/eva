"""ABMIL Network."""

from typing import Type

import torch
import torch.nn as nn

from eva.core.models.networks import MLP


class ABMIL(torch.nn.Module):
    """ABMIL network for multiple instance learning classification tasks.

    Takes an array of patch level embeddings per slide as input. This implementation supports
    batched inputs of shape (`batch_size`, `n_instances`, `input_size`). For slides with less
    than `n_instances` patches, you can apply padding and provide a mask tensor to the forward
    pass.

    The original implementation from [1] was used as a reference:
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py

    Notes:
        - use_bias: The paper didn't use bias in their formalism, but their published
        example code inadvertently does.
        - To prevent dot product similarities near-equal due to concentration of measure
        as a consequence of large input embedding dimensionality (>128), we added the
        option to project the input embeddings to a lower dimensionality

    [1] Maximilian Ilse, Jakub M. Tomczak, Max Welling, "Attention-based Deep Multiple
        Instance Learning", 2018
        https://arxiv.org/abs/1802.04712
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        projected_input_size: int | None,
        hidden_size_attention: int = 128,
        hidden_sizes_mlp: tuple = (128, 64),
        use_bias: bool = True,
        dropout_input_embeddings: float = 0.0,
        dropout_attention: float = 0.0,
        dropout_mlp: float = 0.0,
        pad_value: int | float | None = float("-inf"),
    ) -> None:
        """Initializes the ABMIL network.

        Args:
            input_size: input embedding dimension
            output_size: number of classes
            projected_input_size: size of the projected input. if `None`, no projection is
                performed.
            hidden_size_attention: hidden dimension in attention network
            hidden_sizes_mlp: dimensions for hidden layers in last mlp
            use_bias: whether to use bias in the attention network
            dropout_input_embeddings: dropout rate for the input embeddings
            dropout_attention: dropout rate for the attention network and classifier
            dropout_mlp: dropout rate for the final MLP network
            pad_value: Value indicating padding in the input tensor. If specified, entries with
                this value in the will be masked. If set to `None`, no masking is applied.
        """
        super().__init__()

        self._pad_value = pad_value

        if projected_input_size:
            self.projector = nn.Sequential(
                nn.Linear(input_size, projected_input_size, bias=True),
                nn.Dropout(p=dropout_input_embeddings),
            )
            input_size = projected_input_size
        else:
            self.projector = nn.Dropout(p=dropout_input_embeddings)

        self.gated_attention = GatedAttention(
            input_dim=input_size,
            hidden_dim=hidden_size_attention,
            dropout=dropout_attention,
            n_classes=1,
            use_bias=use_bias,
        )

        self.classifier = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_layer_sizes=hidden_sizes_mlp,
            dropout=dropout_mlp,
            hidden_activation_fn=nn.ReLU,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor: Tensor with expected shape of (batch_size, n_instances, input_size).
        """
        input_tensor, mask = self._mask_values(input_tensor, self._pad_value)

        # (batch_size, n_instances, input_size) -> (batch_size, n_instances, projected_input_size)
        input_tensor = self.projector(input_tensor)

        attention_logits = self.gated_attention(input_tensor)  # (batch_size, n_instances, 1)
        if mask is not None:
            # fill masked values with -inf, which will yield 0s after softmax
            attention_logits = attention_logits.masked_fill(mask, float("-inf"))

        attention_weights = nn.functional.softmax(attention_logits, dim=1)
        # (batch_size, n_instances, 1)

        attention_result = torch.matmul(torch.transpose(attention_weights, 1, 2), input_tensor)
        # (batch_size, 1, hidden_size_attention)

        attention_result = torch.squeeze(attention_result, 1)  # (batch_size, hidden_size_attention)

        return self.classifier(attention_result)  # (batch_size, output_size)

    def _mask_values(self, input_tensor: torch.Tensor, pad_value: float | None):
        """Masks the padded values in the input tensor."""
        if pad_value is None:
            return input_tensor, None
        else:
            # (batch_size, n_instances, input_size)
            mask = input_tensor == pad_value

            # (batch_size, n_instances, input_size) -> (batch_size, n_instances, 1)
            mask = mask.all(dim=-1, keepdim=True)

            # Fill masked values with 0, so that they don't contribute to dense layers
            input_tensor = input_tensor.masked_fill(mask, 0)

            return input_tensor, mask


class GatedAttention(nn.Module):
    """Attention mechanism with Sigmoid Gating using 3 linear layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.25,
        n_classes: int = 1,
        use_bias: bool = True,
        activation_a: Type[nn.Module] = nn.Tanh,
        activation_b: Type[nn.Module] = nn.Sigmoid,
    ):
        """Initializes the GatedAttention network.

        Args:
            input_dim: input feature dimension
            hidden_dim: hidden layer dimension
            dropout: dropout rate
            n_classes: number of classes
            use_bias: whether to use bias in the linear layers
            activation_a: activation function for attention_a.
            activation_b: activation function for attention_b.
        """
        super().__init__()

        def make_attention(activation: nn.Module):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=use_bias), nn.Dropout(p=dropout), activation
            )

        self.attention_a = make_attention(activation_a())
        self.attention_b = make_attention(activation_b())
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        a = self.attention_a(x)  # [..., hidden_dim]
        b = self.attention_b(x)  # [..., hidden_dim]
        att = a.mul(b)  # [..., hidden_dim]
        att = self.attention_c(att)  # [..., n_classes]
        return att
