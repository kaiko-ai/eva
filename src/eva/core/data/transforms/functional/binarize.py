"""The binary threshold process function."""
import torch


def binarize(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Applies the binary threshold function to an input tensor.

    Args:
        tensor: The input tensor.
        threshold: The threshold to binarize the predictions.
    """
    return tensor > threshold
