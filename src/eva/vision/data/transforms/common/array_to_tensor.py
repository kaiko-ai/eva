"""Transformations to convert numpy arrays to torch tensors."""

import numpy.typing as npt
import torch


class ArrayToTensor:
    """Converts a numpy array to a torch tensor."""

    def __call__(self, array: npt.ArrayLike) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            array: The input numpy array.
        """
        return torch.from_numpy(array)


class ArrayToFloatTensor(ArrayToTensor):
    """Converts a numpy array to a torch tensor and casts it to float."""

    def __call__(self, array: npt.ArrayLike):
        """Call method for the transformation.

        Args:
            array: The input numpy array.
        """
        return super().__call__(array).float()
