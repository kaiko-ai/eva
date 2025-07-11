"""Registry for classes and methods."""

from typing import Any, Callable, Dict, List, Type, Union

RegistryItem = Union[Type[Any], Callable[..., Any]]


class Registry:
    """A registry to store and access classes and methods by a unique key."""

    def __init__(self) -> None:
        """Initializes the registry class."""
        self._registry: Dict[str, RegistryItem] = {}

    def register(self, key: str, /) -> Callable[[RegistryItem], RegistryItem]:
        """A decorator to register a class or method with a unique key.

        Args:
            key: The key to register the class or method under.

        Returns:
            A decorator that registers the class or method in the registry.
        """

        def wrapper(obj: RegistryItem) -> RegistryItem:
            if key in self.entries():
                raise ValueError(f"Entry {key} is already registered.")

            self._registry[key] = obj
            return obj

        return wrapper

    def get(self, name: str) -> RegistryItem:
        """Gets the class or method from the registry."""
        if name not in self._registry:
            raise ValueError(f"Item {name} not found in the registry.")
        return self._registry[name]

    def entries(self) -> List[str]:
        """List all items in the registry."""
        return list(self._registry.keys())
