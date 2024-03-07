"""Training related utilities."""

import copy
from collections import abc
from typing import Any


def clone(*inputs: Any) -> Any:
    """Deep copies a list of object and returns them."""
    if not isinstance(inputs, abc.Iterable):
        return copy.deepcopy(inputs)
    return [copy.deepcopy(obj) for obj in inputs]
