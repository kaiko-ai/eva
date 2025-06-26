"""Transforms for extracting the CLS output from a model output."""

import torch
from transformers import modeling_outputs


class ExtractCLSFeatures:
    """Extracts the CLS token from a ViT model output."""

    def __init__(
        self, cls_index: int = 0, num_register_tokens: int = 0, include_patch_tokens: bool = False
    ) -> None:
        """Initializes the transformation.

        Args:
            cls_index: The index of the CLS token in the output tensor.
            num_register_tokens: The number of register tokens in the model output.
            include_patch_tokens: Whether to concat the mean aggregated patch tokens with
                the cls token.
        """
        self._cls_index = cls_index
        self._num_register_tokens = num_register_tokens
        self._include_patch_tokens = include_patch_tokens

    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The tensor representing the model output.
        """
        if isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            tensor = tensor.last_hidden_state  # type: ignore

        cls_token = tensor[:, self._cls_index, :]
        if self._include_patch_tokens:
            patch_tokens = tensor[:, 1 + self._num_register_tokens :, :]
            return torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)

        return cls_token
