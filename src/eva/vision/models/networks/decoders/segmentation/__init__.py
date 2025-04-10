"""Segmentation decoder heads API."""

from eva.vision.models.networks.decoders.segmentation.base import Decoder
from eva.vision.models.networks.decoders.segmentation.decoder2d import Decoder2D
from eva.vision.models.networks.decoders.segmentation.linear import LinearDecoder
from eva.vision.models.networks.decoders.segmentation.semantic import (
    ConvDecoder1x1,
    ConvDecoderMS,
    ConvDecoderWithImage,
    SingleLinearDecoder,
    SwinUNETRDecoder,
)

__all__ = [
    "Decoder",
    "Decoder2D",
    "ConvDecoder1x1",
    "ConvDecoderMS",
    "ConvDecoderWithImage",
    "LinearDecoder",
    "SingleLinearDecoder",
    "SwinUNETRDecoder",
]
