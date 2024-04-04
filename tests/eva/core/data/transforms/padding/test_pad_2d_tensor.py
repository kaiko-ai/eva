"""Tests for Pad2DTensor transformation."""

from typing import Tuple

import torch

from eva.core.data.transforms import Pad2DTensor


def test_no_padding_needed():
    """Test case where tensor larger than pad size and no padding is needed."""
    pad_size = 5
    embedding_dim = 10
    tensor = create_tensor((6, embedding_dim))
    transformation = Pad2DTensor(pad_size=pad_size)
    result_tensor = transformation(tensor)
    assert result_tensor.shape == tensor.shape, "No padding should be applied"


def test_padding_needed():
    """Test case where tensor smaller than pad size and padding should be applied."""
    pad_size = 5
    embedding_dim = 10
    tensor = create_tensor((3, embedding_dim))  # tensor smaller than pad size
    transformation = Pad2DTensor(pad_size=pad_size)
    result_tensor = transformation(tensor)
    assert result_tensor.shape == (
        pad_size,
        embedding_dim,
    ), "Padding should be applied to match the pad_size"


def test_padding_value():
    """Test if padded entries have the correct value."""
    pad_size = 5
    embedding_dim = 10
    tensor = create_tensor((3, embedding_dim))
    pad_value = 0
    transformation = Pad2DTensor(pad_size=pad_size, pad_value=pad_value)
    result_tensor = transformation(tensor)
    # Check if padding values are as expected
    assert (
        result_tensor[3:] == pad_value
    ).all(), "Padded values should match the specified pad_value"


def create_tensor(shape: Tuple[int, ...], value: float = 1.0):
    """Helper function to create a tensor."""
    return torch.full(shape, fill_value=value, dtype=torch.float32)
