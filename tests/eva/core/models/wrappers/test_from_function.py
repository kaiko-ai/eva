"""ModelFromFunction tests."""

from typing import Any, Callable, Dict

import pytest
import torch
from torch import nn

from eva.core.models import wrappers


@pytest.mark.parametrize(
    "path, arguments",
    [
        (torch.nn.Flatten, None),
        (torch.nn.Linear, {"in_features": 10, "out_features": 2}),
    ],
)
def test_model_from_function(
    model_from_function: wrappers.ModelFromFunction,
) -> None:
    """Tests the model_from_function network."""
    input_tensor = torch.Tensor(4, 10)
    output = model_from_function(input_tensor)
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize(
    "path, arguments",
    [
        (torch.nn.Flatten, {"invalid_arg": 1}),
        (torch.nn.Linear, {"invalid_arg": 12}),
    ],
)
def test_error_model_from_function(
    path: Callable[..., nn.Module],
    arguments: Dict[str, Any] | None,
) -> None:
    """Tests the model_from_function network."""
    with pytest.raises(TypeError):
        wrappers.ModelFromFunction(path=path, arguments=arguments)


@pytest.fixture(scope="function")
def model_from_function(
    path: Callable[..., nn.Module],
    arguments: Dict[str, Any] | None,
) -> wrappers.ModelFromFunction:
    """ModelFromFunction fixture."""
    return wrappers.ModelFromFunction(path=path, arguments=arguments)
