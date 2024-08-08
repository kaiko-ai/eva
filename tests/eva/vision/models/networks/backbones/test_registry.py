"""Vision Model Registry tests."""

from typing import Callable

import pytest
import torch.nn as nn

from eva.vision.models.networks.backbones import BackboneModelRegistry, register_model


@pytest.mark.parametrize("model_name", ["universal/vit_small_patch16_224_random"])
def test_get_model(model_name: str):
    """Test getting a model function from the registry."""
    model_fn = BackboneModelRegistry.get(model_name)
    assert isinstance(model_fn, Callable)


@pytest.mark.parametrize("model_name", ["universal/vit_small_patch16_224_random"])
def test_load_model(model_name: str):
    """Test loading an instantiated model from the registry."""
    model = BackboneModelRegistry.load_model(model_name)
    assert isinstance(model, nn.Module)


def test_list_models():
    """Test listing all models in the registry."""
    models = BackboneModelRegistry.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_register_model():
    """Test registering and loading a model."""
    model_name = "test_model_1"
    with pytest.raises(ValueError, match=f"Model {model_name} not found in the registry."):
        BackboneModelRegistry.get(model_name)

    @register_model(model_name)
    def dummy_model_fn():
        return DummyModel()

    model_fn = BackboneModelRegistry.get(model_name)
    assert isinstance(model_fn, Callable)
    assert isinstance(model_fn(), DummyModel)


def test_register_model_duplicate():
    """Test if registry raises error when the model name already exists."""
    model_name = "test_model_2"

    @register_model(model_name)
    def dummy_model_fn():
        return DummyModel()

    with pytest.raises(ValueError, match=f"Model {model_name} is already registered."):

        @register_model(model_name)
        def dummy_model_fn():
            return DummyModel()


class DummyModel(nn.Module):
    """Dummy model."""

    def __init__(self):
        """Initializes the dummy model."""
        super().__init__()
