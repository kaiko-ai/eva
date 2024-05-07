"""Dataset related function and helper functions."""

from typing import List, Sequence, Tuple


def indices_to_ranges(indices: List[int]) -> List[Tuple[int, int]]:
    """Turns a list of indices to a list of ranges.

    The produced range intervals are half-open inequalities: start <= x < end.

    Args:
        indices: The list of indices to produce the ranges from.

    Return:
        A list of half-open intervals.

    Example:
        >>> indices = [0, 1, 2, 4, 6, 7, 8]
        >>> ranges = indices_to_ranges(indices)
        >>> assert ranges == [(0, 3), (4, 5), (6, 9)]
    """
    ranges = []
    start_index = 0
    for i, current in enumerate(indices):
        if i + 1 < len(indices) and current + 1 == indices[i + 1]:
            continue

        start = indices[start_index]
        end = start if start_index == i else current
        ranges.append((start, end + 1))
        start_index = i + 1

    return ranges


def ranges_to_indices(ranges: Sequence[Tuple[int, int]]) -> List[int]:
    """Unpacks a list of ranges to individual indices.

    Args:
        ranges: A sequence of ranges to produce the indices from.

    Return:
        A list of the indices.

    Example:
        >>> ranges == [(0, 3), (4, 5), (6, 9)]
        >>> indices = ranges_to_indices(ranges)
        >>> assert indices == [0, 1, 2, 4, 6, 7, 8]
    """
    return [index for start, end in ranges for index in range(start, end)]
