"""Transforms for extracting the patch features from a model output."""

import math
from typing import List

import torch
from transformers import modeling_outputs


class ExtractPatchFeatures:
    """Extracts the patch features from a ViT model output."""

    def __call__(
        self, tensor: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> List[torch.Tensor]:
        """Call method for the transformation.

        Args:
            tensor: The raw embeddings of the model.

        Returns:
            A tensor (batch_size, hidden_size, n_patches_height, n_patches_width)
            representing the model output.
        """
        if isinstance(tensor, modeling_outputs.BaseModelOutputWithPooling):
            features = tensor.last_hidden_state[:, 1:, :].permute(0, 2, 1)
            batch_size, hidden_size, patch_grid = features.shape
            height = width = int(math.sqrt(patch_grid))
            patch_embeddings = features.view(batch_size, hidden_size, height, width)
        else:
            raise ValueError(f"Unsupported type {type(tensor)}")

        return [patch_embeddings]