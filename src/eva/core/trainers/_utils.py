"""Training related utilities."""

import copy
from typing import Any


def clone(*inputs: Any) -> Any:
    """Deep copies a list of object and returns them."""
    if len(inputs) == 1:
        return copy.deepcopy(inputs[0])
    return [copy.deepcopy(obj) for obj in inputs]
