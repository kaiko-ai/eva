"""Transforms for extracting the CLS output from a model output."""

import torch
from transformers import modeling_outputs


class ExtractCLSFeatures:
    """Extracts the CLS token from a ViT model output."""

    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The tensor representing the model output.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor[:, 0, :]
        elif isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            return tensor.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unsupported type {type(tensor)}")
