import torch
from transformers import modeling_outputs


class ExtractCLSFeatures:
    """Extracts the CLS token from a ViT model output."""

    def __call__(
        self, x: torch.Tensor | modeling_outputs.BaseModelOutputWithPooling
    ) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            x: The tensor representing the model output.
        """
        if isinstance(x, torch.Tensor):
            return x[:, 0, :]
        elif isinstance(x, modeling_outputs.BaseModelOutputWithPooling):
            return x.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unsupported type {type(x)}")
