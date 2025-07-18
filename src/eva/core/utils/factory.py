"""Factory classes."""

import inspect
from typing import Any, Dict, Generic, Type, TypeVar

from torch import nn
from typing_extensions import override

from eva.core.utils.registry import Registry, RegistryItem

T = TypeVar("T")


class Factory(Generic[T]):
    """A base factory class for instantiating registry items of a specific type."""

    def __new__(cls, registry: Registry, name: str, init_args: dict, expected_type: Type[T]) -> T:
        """Creates the appropriate instance based on registry entry.

        Args:
            registry: The registry containing the items to instantiate.
            name: Name of the registry item to instantiate.
            init_args: The arguments to pass to the constructor of the registry item.
            expected_type: The expected type of the instantiated object.
        """
        if name not in registry.entries():
            raise ValueError(
                f"Invalid name: {name}. Please choose one "
                f"of the following: {registry.entries()}"
            )

        registry_item = registry.get(name)
        filtered_kwargs = _filter_kwargs(registry_item, init_args)
        instance = registry_item(**filtered_kwargs)

        if not isinstance(instance, expected_type):
            raise TypeError(f"Expected an instance of {expected_type}, but got {type(instance)}.")
        return instance


class ModuleFactory(Factory[nn.Module]):
    """Factory class for instantiating nn.Module instances from a registry."""

    @override
    def __new__(cls, registry: Registry, name: str, init_args: dict) -> nn.Module:
        return super().__new__(cls, registry, name, init_args, nn.Module)


def _filter_kwargs(registry_item: RegistryItem, kwargs: dict) -> Dict[str, Any]:
    """Filters the given keyword arguments to match the signature of a given class or method.

    Args:
        registry_item: The class or method from the registry whose
            signature should be used for filtering.
        kwargs: The keyword arguments to filter.

    Returns:
        A dictionary containing only the valid keyword arguments that match
        the callable's parameters.
    """
    if inspect.isclass(registry_item):
        signature = inspect.signature(registry_item.__init__)
    else:
        signature = inspect.signature(registry_item)

    return {k: v for k, v in kwargs.items() if k in signature.parameters}
