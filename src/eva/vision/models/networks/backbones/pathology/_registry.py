from typing import Any, Callable, Dict, Type

import torch.nn as nn


class PathologyModelRegistry:
    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._registry:
                raise ValueError(f"Model {name} is already registered.")
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        if name not in cls._registry:
            raise ValueError(f"Model {name} not found in the registry.")
        return cls._registry[name]

    @staticmethod
    def load_model(model_name: str, **kwargs: Any) -> nn.Module:
        model_class = PathologyModelRegistry.get(model_name)
        return model_class(**kwargs)
