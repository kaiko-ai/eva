"""Tests for linear decoder."""

from typing import List, Tuple

import pytest
import torch
from torch import nn

from eva.vision.models.networks import decoders


@pytest.mark.parametrize(
    "layers, features, image_size, expected_shape",
    [
        (
            nn.Linear(384, 5),
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
    ],
)
def test_linear_decoder(
    linear_decoder: decoders.LinearDecoder,
    features: List[torch.Tensor],
    image_size: Tuple[int, int],
    expected_shape: torch.Size,
) -> None:
    """Tests the ConvDecoder network."""
    logits = linear_decoder(features, image_size)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == expected_shape


@pytest.fixture(scope="function")
def linear_decoder(
    layers: nn.Module,
) -> decoders.LinearDecoder:
    """LinearDecoder fixture."""
    return decoders.LinearDecoder(layers=layers)
