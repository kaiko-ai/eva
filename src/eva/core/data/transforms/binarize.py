import torch

from eva.core.data.transforms import functional


class Binarize:
    """Applies the binarization function."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initializes the process.

        Args:
            threshold: The threshold to binarize the predictions.
        """
        super().__init__()

        self._threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return functional.binarize(tensor, self._threshold)
