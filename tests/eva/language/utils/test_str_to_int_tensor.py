"""Tests the CastStrToIntTensor transform."""

import pytest
import torch

from eva.language.utils.str_to_int_tensor import CastStrToIntTensor


@pytest.mark.parametrize(
    "input_values, expected",
    [
        ("0", torch.tensor([0], dtype=torch.int)),
        (["0", "1", "2"], torch.tensor([0, 1, 2], dtype=torch.int)),
        ([0, 1, 2], torch.tensor([0, 1, 2], dtype=torch.int)),
        ([], torch.tensor([], dtype=torch.int)),
    ],
)
def test_cast_str_to_int_tensor_valid(input_values, expected):
    """Test CastStrToIntTensor with valid inputs."""
    result = CastStrToIntTensor()(input_values)
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "invalid_input",
    ["abc", ["1", "abc"], None],
)
def test_cast_str_to_int_tensor_invalid(invalid_input):
    """Test CastStrToIntTensor with invalid inputs."""
    with pytest.raises(ValueError):
        CastStrToIntTensor()(invalid_input)
