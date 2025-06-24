"""Tests for convolutional decoder."""

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
            nn.Conv2d(384, 5, kernel_size=(1, 1)),
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(384, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 5, kernel_size=(3, 3), padding=(1, 1)),
            ),
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            nn.Conv2d(768, 5, kernel_size=(1, 1)),
            [torch.Tensor(2, 384, 14, 14), torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            common.ConvDecoder1x1(384, 5)._layers,
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
        (
            common.ConvDecoderMS(384, 5)._layers,
            [torch.Tensor(2, 384, 14, 14)],
            (224, 224),
            torch.Size([2, 5, 224, 224]),
        ),
    ],
)
def test_conv_decoder(
    conv_decoder: segmentation.Decoder2D,
    features: List[torch.Tensor],
    image_size: Tuple[int, int],
    expected_shape: torch.Size,
) -> None:
    """Tests the ConvDecoder network."""
    inputs = DecoderInputs(features, image_size)
    logits = conv_decoder(inputs)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == expected_shape


@pytest.fixture(scope="function")
def conv_decoder(
    layers: nn.Module,
) -> segmentation.Decoder2D:
    """ConvDecoder fixture."""
    return segmentation.Decoder2D(layers=layers)
