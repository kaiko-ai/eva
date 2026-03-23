"""Functional operations."""

import re
from typing import Iterable, List


def numeric_sort(item: Iterable[str], /) -> List[str]:
    """Sorts an iterable of strings treating embedded numbers as numeric values.

    Here the strings are compared based on their numeric value rather than their
    string representation.

    Args:
        item: An iterable of strings to be sorted.

    Returns:
        A list of strings sorted based on their numeric values.
    """
    return sorted(
        item,
        key=lambda value: re.sub(
            r"(\d+)",
            lambda num: f"{int(num[0]):010d}",
            value,
        ),
    )
