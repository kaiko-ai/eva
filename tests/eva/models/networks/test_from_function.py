"""ModelFromFunction tests."""

from typing import Any, Callable, Dict

import pytest
import torch
from torch import nn

from eva.models import networks

INPUT_TENSOR = torch.Tensor(4, 10)

PATH_A = torch.nn.Flatten
ARGS_A = None

PATH_B = torch.nn.Linear
ARGS_B = {
    "in_features": 10,
    "out_features": 2,
}
"""Test features."""


@pytest.mark.parametrize(
    "path, arguments",
    [
        (PATH_A, ARGS_A),
        (PATH_B, ARGS_B),
    ],
)
def test_model_from_function(
    model_from_function: networks.ModelFromFunction,
) -> None:
    """Tests the model_from_function network."""
    output = model_from_function(INPUT_TENSOR)
    assert isinstance(output, torch.Tensor)


@pytest.mark.parametrize(
    "path, arguments",
    [
        (PATH_A, {"invalid_arg": 1}),
        (PATH_B, {"invalid_arg": "2"}),
    ],
)
def test_error_model_from_function(
    path: Callable[..., nn.Module],
    arguments: Dict[str, Any] | None,
) -> None:
    """Tests the model_from_function network."""
    with pytest.raises(TypeError):
        networks.ModelFromFunction(path=path, arguments=arguments)


@pytest.fixture(scope="function")
def model_from_function(
    path: Callable[..., nn.Module],
    arguments: Dict[str, Any] | None,
) -> networks.ModelFromFunction:
    """ModelFromFunction fixture."""
    return networks.ModelFromFunction(path=path, arguments=arguments)
