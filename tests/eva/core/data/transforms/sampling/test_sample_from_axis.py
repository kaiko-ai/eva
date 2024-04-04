"""Tests for SampleFromAxis transformation."""

from typing import Tuple

import torch

from eva.core.data.transforms import SampleFromAxis


def test_correct_number_of_samples():
    """Ensure the sampled tensor has the correct number of samples."""
    tensor = create_tensor((10, 5))  # Create a 10x5 tensor
    transformation = SampleFromAxis(n_samples=3)
    sampled_tensor = transformation(tensor)
    assert sampled_tensor.shape == (3, 5), "Output tensor should have 3 samples along the 0th axis"


def test_consistency_with_seed():
    """Check if the same seed results in the same sampled output."""
    tensor = create_tensor((10, 5))
    transformation1 = SampleFromAxis(n_samples=3, seed=42)
    transformation2 = SampleFromAxis(n_samples=3, seed=42)
    assert torch.equal(
        transformation1(tensor), transformation2(tensor)
    ), "Sampling with the same seed should produce identical results"


def test_axis_sampling():
    """Verify that sampling occurs along the correct axis."""
    tensor = create_tensor((5, 10))  # Create a 5x10 tensor
    transformation = SampleFromAxis(n_samples=3, axis=1)
    sampled_tensor = transformation(tensor)
    assert sampled_tensor.shape == (5, 3), "Output tensor should have 3 samples along the 1st axis"


def test_n_samples_greater_than_dimension():
    """Ensure correct behavior when n_samples exceeds the axis dimension."""
    tensor = create_tensor((5, 10))
    transformation = SampleFromAxis(
        n_samples=10, axis=0
    )  # Request more samples than available in axis 0
    sampled_tensor = transformation(tensor)
    assert sampled_tensor.shape == (
        5,
        10,
    ), "Output tensor should remain unchanged when n_samples exceeds axis dimension"


def test_zero_samples():
    """Test the behavior when n_samples is set to zero."""
    tensor = create_tensor((10, 5))
    transformation = SampleFromAxis(n_samples=0)
    sampled_tensor = transformation(tensor)
    assert (
        sampled_tensor.numel() == 0
    ), "Output tensor should have zero elements when n_samples is zero"


def test_single_dimension_tensor():
    """Check sampling from a tensor with only one dimension."""
    tensor = create_tensor((10,))  # Create a tensor with only one dimension
    transformation = SampleFromAxis(n_samples=3)
    sampled_tensor = transformation(tensor)
    assert sampled_tensor.shape == (
        3,
    ), "Output tensor should correctly sample from a single-dimension tensor"


def create_tensor(shape: Tuple[int, ...], value: float = 1.0):
    """Helper function to create a tensor."""
    return torch.full(shape, fill_value=value, dtype=torch.float32)
