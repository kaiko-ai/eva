"""Backbone Model Registry."""

from typing import Any, Callable, Dict, List

import torch.nn as nn


class BackboneModelRegistry:
    """A model registry for accessing backbone models by name."""

    _registry: Dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a new model."""

        def decorator(model_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            if name in cls._registry:
                raise ValueError(f"Model {name} is already registered.")
            cls._registry[name] = model_fn
            return model_fn

        return decorator

    @classmethod
    def get(cls, model_name: str) -> Callable[..., nn.Module]:
        """Gets a model function from the registry."""
        if model_name not in cls._registry:
            raise ValueError(f"Model {model_name} not found in the registry.")
        return cls._registry[model_name]

    @classmethod
    def load_model(cls, model_name: str, model_kwargs: Dict[str, Any] | None = None) -> nn.Module:
        """Loads & initializes a model class from the registry."""
        model_fn = cls.get(model_name)
        return model_fn(**(model_kwargs or {}))

    @classmethod
    def list_models(cls) -> List[str]:
        """List all models in the registry."""
        register_models = [name for name in cls._registry.keys() if not name.startswith("timm")]
        return register_models + ["timm/<model_name>"]


def register_model(name: str) -> Callable:
    """Simple decorator to register a model."""
    return BackboneModelRegistry.register(name)
