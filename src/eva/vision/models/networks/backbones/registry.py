"""Backbone Model Registry."""

from typing import Any, Callable, Dict, Type

import torch.nn as nn


class BackboneModelRegistry:
    """A model registry for accessing backbone models by name."""

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a new model."""

        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._registry:
                raise ValueError(f"Model {name} is already registered.")
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[..., nn.Module]:
        """Gets a model function from the registry."""
        if name not in cls._registry:
            raise ValueError(f"Model {name} not found in the registry.")
        return cls._registry[name]

    @classmethod
    def load_model(cls, model_name: str, **kwargs: Any) -> nn.Module:
        """Loads & initializes a model class from the registry."""
        model_fct = cls.get(model_name)
        return model_fct(**kwargs)
