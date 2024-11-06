"""Squeeze transform."""

import numpy as np
import numpy.typing as npt


class ArraySqueeze:
    """Returns a array with all specified dimensions of input of size 1 removed."""

    def __call__(self, array: npt.NDArray) -> npt.NDArray:
        """Call method for the transformation.

        Args:
            array: The input array.

        Returns:
            A squeezed version of the input array.
        """
        return np.squeeze(array)
