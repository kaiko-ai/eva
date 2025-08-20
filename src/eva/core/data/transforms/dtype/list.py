"""Transformations to convert lists."""

from typing import List

import torch


class ListToTensor:
    """Converts a list to a torch tensor."""

    def __call__(self, data: List[int | float | str]) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            data: A list of values to be converted to a tensor.
        """
        return torch.tensor(data)
