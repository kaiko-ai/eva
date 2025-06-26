"""Tests the CastStrToIntTensor transform."""

import pytest
import torch

from eva.language.utils.str_to_int_tensor import CastStrToIntTensor


@pytest.mark.parametrize(
    "input_values, expected",
    [
        # Test with text responses (default mapping)
        ("yes", torch.tensor([1], dtype=torch.int)),
        ("no", torch.tensor([0], dtype=torch.int)),
        ("maybe", torch.tensor([2], dtype=torch.int)),
        (["no", "yes", "maybe"], torch.tensor([0, 1, 2], dtype=torch.int)),
        # Test numeric fallback
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


@pytest.mark.parametrize(
    "custom_mapping, input_values, expected",
    [
        (
            {r"positive|good": 1, r"negative|bad": 0},
            ["positive", "bad"],
            torch.tensor([1, 0], dtype=torch.int),
        ),
        (
            {r"positive|good": 1, r"negative|bad": 0},
            ["good", "negative"],
            torch.tensor([1, 0], dtype=torch.int),
        ),
        (
            {r"positive|good": 1, r"negative|bad": 0},
            ["POSITIVE", "BAD"],
            torch.tensor([1, 0], dtype=torch.int),
        ),
        (
            {r"\bhappy\b": 1, r"\bsad\b": 0},
            ["I am happy", "feeling sad"],
            torch.tensor([1, 0], dtype=torch.int),
        ),
    ],
)
def test_cast_str_to_int_tensor_custom_mapping(custom_mapping, input_values, expected):
    """Test CastStrToIntTensor with custom mapping."""
    transform = CastStrToIntTensor(custom_mapping)
    result = transform(input_values)
    assert torch.equal(result, expected)
