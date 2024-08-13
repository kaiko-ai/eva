"""Transforms for extracting the CLS output from a model output."""

import torch
from transformers import modeling_outputs


class ExtractCLSFeatures:
    """Extracts the CLS token from a ViT model output."""

    def __init__(self, cls_index: int = 0) -> None:
        """Initializes the transformation.

        Args:
            cls_index: The index of the CLS token in the output tensor.
        """
        self._cls_index = cls_index

    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The tensor representing the model output.
        """
        if isinstance(tensor, torch.Tensor):
            transformed_tensor = tensor[:, self._cls_index, :]
        elif isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            transformed_tensor = tensor.last_hidden_state[:, self._cls_index, :]
        else:
            raise ValueError(f"Unsupported type {type(tensor)}")

        return transformed_tensor
