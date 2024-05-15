import torch

from eva.core.data.transforms import functional


class Argmax:
    """Applies the binarization function."""

    def __init__(self, dim: int = 1) -> None:
        """Initializes the process.

        Args:
            threshold: The threshold to binarize the predictions.
        """
        super().__init__()

        self._dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.argmax(tensor, dim=self._dim)
