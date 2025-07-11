"""Tests for the factory module."""

import pytest
from torch import nn

from eva.core.utils import factory
from eva.core.utils.registry import Registry


class _MockModel(nn.Module):
    def __init__(self, param1: int = 1, param2: str = "default"):
        super().__init__()
        self.param1 = param1
        self.param2 = param2


class _MockClass:
    def __init__(self, arg1: int, arg2: str = "test"):
        self.arg1 = arg1
        self.arg2 = arg2


def _mock_function(x: int, y: str = "default") -> str:
    return f"{x}_{y}"


@pytest.fixture
def test_registry():
    """Fixture to create a test registry with mock items."""
    registry = Registry()
    registry.register("mock_model")(_MockModel)
    registry.register("mock_class")(_MockClass)
    registry.register("mock_function")(_mock_function)
    return registry


def test_factory_valid_instantiation(test_registry):
    """Test Factory creates valid instances."""
    instance = factory.Factory(test_registry, "mock_class", {"arg1": 42}, _MockClass)
    assert isinstance(instance, _MockClass)
    assert instance.arg1 == 42
    assert instance.arg2 == "test"


def test_factory_with_filtered_kwargs(test_registry):
    """Test Factory filters kwargs correctly."""
    init_args = {"arg1": 100, "arg2": "custom", "invalid_arg": "ignored"}
    instance = factory.Factory(test_registry, "mock_class", init_args, _MockClass)
    assert instance.arg1 == 100
    assert instance.arg2 == "custom"


def test_factory_invalid_name(test_registry):
    """Test Factory raises ValueError for invalid name."""
    with pytest.raises(ValueError, match="Invalid name: nonexistent"):
        factory.Factory(test_registry, "nonexistent", {}, _MockClass)


def test_factory_type_mismatch(test_registry):
    """Test Factory raises TypeError for type mismatch."""
    with pytest.raises(TypeError, match="Expected an instance of"):
        factory.Factory(test_registry, "mock_function", {"x": 1}, _MockClass)


def test_module_factory_valid_instantiation(test_registry):
    """Test ModuleFactory creates valid nn.Module instances."""
    instance = factory.ModuleFactory(test_registry, "mock_model", {"param1": 5})
    assert isinstance(instance, nn.Module)
    assert isinstance(instance, _MockModel)
    assert instance.param1 == 5
    assert instance.param2 == "default"


def test_filter_kwargs_class():
    """Test _filter_kwargs with class constructor."""
    kwargs = {"arg1": 42, "arg2": "test", "invalid": "ignored"}
    filtered = factory._filter_kwargs(_MockClass, kwargs)
    assert filtered == {"arg1": 42, "arg2": "test"}


def test_filter_kwargs_function():
    """Test _filter_kwargs with function."""
    kwargs = {"x": 1, "y": "hello", "z": "ignored"}
    filtered = factory._filter_kwargs(_mock_function, kwargs)
    assert filtered == {"x": 1, "y": "hello"}
