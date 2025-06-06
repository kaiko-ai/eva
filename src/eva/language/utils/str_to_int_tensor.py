"""Transform utilities for post-processing predictions."""

from typing import Any, List, Union

import torch


class CastStrToIntTensor:
    """Casts string predictions like ['0', '1', '2'] to a torch.Tensor of ints.

    This transform is useful when model outputs or predictions are strings or
    stringified numbers (e.g., '0', '1') and need to be converted into integer
    tensors for evaluation or further computation.

    Supports single values, lists of strings, or lists of integers.

    Example:
        >>> transform = CastStrToIntTensor()
        >>> transform(['0', '1', '2'])
        tensor([0, 1, 2])
        >>> transform('3')
        tensor([3])
    """

    def __call__(self, values: Union[str, List[str], List[int]]) -> torch.Tensor:
        """Convert string or list of strings/ints to a torch.Tensor of integers.

        Args:
            values: A string, or a list of strings/integers representing numeric values.

        Returns:
            A 1D torch.Tensor of integers.

        Raises:
            ValueError: If any value in the input cannot be cast to an integer.
        """
        return torch.tensor(
            [self._cast_single(v) for v in (values if isinstance(values, list) else [values])],
            dtype=torch.int,
        )

    def _cast_single(self, value: Any) -> int:
        """Casts a single value to an integer.

        Args:
            value: A single value to convert (typically a string or int).

        Returns:
            The value as an integer.

        Raises:
            ValueError: If the value cannot be converted to an integer.
        """
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert value to int: {value!r}") from e
