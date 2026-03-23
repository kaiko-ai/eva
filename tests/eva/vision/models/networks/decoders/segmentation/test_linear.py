"""Tests for linear decoder."""

from typing import List, Tuple

import pytest
import torch
from torch import nn

from eva.vision.models.networks.decoders import segmentation
from eva.vision.models.networks.decoders.segmentation.semantic import common
from eva.vision.models.networks.decoders.segmentation.typings import DecoderInputs


@pytest.mark.parametrize(
    "layers, features, image_size, expected_shape",
    [
        (
            nn.Linear(384, 5),
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            nn.Linear(768, 5),
            [torch.Tensor(2, 384, 14, 14), torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            common.SingleLinearDecoder(384, 5)._layers,
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            common.SingleLinearDecoder(768, 5)._layers,
            [torch.Tensor(2, 384, 14, 14), torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
    ],
)
def test_linear_decoder(
    linear_decoder: segmentation.LinearDecoder,
    features: List[torch.Tensor],
    image_size: Tuple[int, int],
    expected_shape: torch.Size,
) -> None:
    """Tests the ConvDecoder network."""
    inputs = DecoderInputs(features=features, image_size=image_size)
    logits = linear_decoder(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == expected_shape


@pytest.fixture(scope="function")
def linear_decoder(
    layers: nn.Module,
) -> segmentation.LinearDecoder:
    """LinearDecoder fixture."""
    return segmentation.LinearDecoder(layers=layers)
