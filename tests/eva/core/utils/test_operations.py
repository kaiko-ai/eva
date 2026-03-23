"""Tests the functional operations."""

from typing import List

import pytest

from eva.core.utils import operations


@pytest.mark.parametrize(
    "input_list, expected",
    [
        (["a10", "a2", "a1"], ["a1", "a2", "a10"]),
        (["item20", "item3", "item1", "item10"], ["item1", "item3", "item10", "item20"]),
        (["b", "a", "c"], ["a", "b", "c"]),
        ([], []),
        (["a1b2", "a1b10", "a1b3"], ["a1b2", "a1b3", "a1b10"]),
        (["a2", "b1", "a10", "b2"], ["a2", "a10", "b1", "b2"]),
        (
            ["item0020", "item0003", "item0001", "item0010"],
            ["item0001", "item0003", "item0010", "item0020"],
        ),
    ],
)
def test_numeric_sort(input_list: List[str], expected: List[str]) -> None:
    """Test numeric_sort function with various input cases."""
    assert operations.numeric_sort(input_list) == expected
