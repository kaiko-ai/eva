"""Transform utilities for post-processing predictions."""

import re
from typing import Any, Dict, List, Union

import torch


class CastStrToIntTensor:
    """Casts string predictions to a torch.Tensor of ints using regex mapping.

    This transform is useful when model outputs are text responses (e.g., 'yes', 'no', 'maybe')
    that need to be converted into integer tensors for evaluation. It uses regex patterns
    to map text responses to integer labels, making it flexible for various classification tasks.

    Supports single values, lists of strings, or lists of integers.

    Example:
        >>> # Default mapping for yes/no/maybe classification
        >>> transform = CastStrToIntTensor()
        >>> transform(['yes', 'no', 'maybe'])
        tensor([1, 0, 2])
        >>> transform('yes')
        tensor([1])

        >>> # Custom mapping
        >>> transform = CastStrToIntTensor({r'positive|good': 1, r'negative|bad': 0})
        >>> transform(['positive', 'bad'])
        tensor([1, 0])
    """

    def __init__(self, mapping: Dict[str, int] | None = None):
        """Initialize the transform with a regex-to-integer mapping.

        Args:
            mapping: Dictionary mapping regex patterns to integers. If None, uses default
                    yes/no/maybe mapping: {'no': 0, 'yes': 1, 'maybe': 2}
        """
        if mapping is None:
            self.mapping = {r"\bno\b": 0, r"\byes\b": 1, r"\bmaybe\b": 2}
        else:
            self.mapping = mapping

        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), value) for pattern, value in self.mapping.items()
        ]

    def __call__(self, values: Union[str, List[str], List[int]]) -> torch.Tensor:
        """Convert string or list of strings/ints to a torch.Tensor of integers.

        Args:
            values: A string, or a list of strings/integers representing responses.

        Returns:
            A 1D torch.Tensor of integers.

        Raises:
            ValueError: If any value cannot be mapped to an integer.
        """
        return torch.tensor(
            [self._cast_single(v) for v in (values if isinstance(values, list) else [values])],
            dtype=torch.int,
        )

    def _cast_single(self, value: Any) -> int:
        """Casts a single value to an integer using regex mapping.

        Args:
            value: A single value to convert (typically a string or int).

        Returns:
            The value as an integer.

        Raises:
            ValueError: If the value cannot be mapped to an integer.
        """
        if isinstance(value, int):
            return value

        if not isinstance(value, str):
            value = str(value)

        value = value.strip()

        for pattern, mapped_value in self.compiled_patterns:
            if pattern.search(value):
                return mapped_value

        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot map value to int: {value!r}. "
                f"Available patterns: {list(self.mapping.keys())}"
            ) from e
