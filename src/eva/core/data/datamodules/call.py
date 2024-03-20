"""Helper dataset calling methods."""

from typing import Any, Iterable

from eva.core.data import datasets as datasets_lib


def call_method_if_exists(objects: Iterable[Any], /, method: str) -> None:
    """Calls a desired `method` from the datasets if exists.

    Args:
        objects: An iterable of objects.
        method: The dataset method name to call if exists.
    """
    for _object in _recursive_iter(objects):
        if hasattr(_object, method):
            fn = getattr(_object, method)
            fn()


def _recursive_iter(objects: Iterable[Any], /) -> Iterable[datasets_lib.TorchDataset]:
    """Iterates through an iterable of objects and their respective iterable values.

    Args:
        objects: The objects to iterate from.

    Yields:
        The individual object class.
    """
    for _object in objects:
        if not isinstance(_object, list):
            _object = [_object]
        yield from _object
