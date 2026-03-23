"""Vision Model Registry tests."""

from typing import Callable

import pytest
import torch.nn as nn

from eva.vision.models.networks.backbones import backbone_registry


@pytest.mark.parametrize("model_name", ["universal/vit_small_patch16_224_random"])
def test_get_model(model_name: str):
    """Test getting a model function from the registry."""
    model_fn = backbone_registry.get(model_name)
    assert isinstance(model_fn, Callable)


def test_list_entries():
    """Test listing all entries in the registry."""
    models = backbone_registry.entries()
    assert isinstance(models, list)
    assert len(models) > 0


def test_register_model():
    """Test registering and loading a model."""
    model_name = "test_model_1"
    with pytest.raises(ValueError, match=f"Item {model_name} not found in the registry."):
        backbone_registry.get(model_name)

    @backbone_registry.register(model_name)
    def dummy_model_fn():
        return DummyModel()

    model_fn = backbone_registry.get(model_name)
    assert isinstance(model_fn, Callable)
    assert isinstance(model_fn(), DummyModel)


def test_register_model_duplicate():
    """Test if registry raises error when the model name already exists."""
    model_name = "test_model_2"

    @backbone_registry.register(model_name)
    def dummy_model_fn():
        return DummyModel()

    with pytest.raises(ValueError, match=f"Entry {model_name} is already registered."):

        @backbone_registry.register(model_name)
        def dummy_model_fn():
            return DummyModel()


class DummyModel(nn.Module):
    """Dummy model."""

    def __init__(self):
        """Initializes the dummy model."""
        super().__init__()
