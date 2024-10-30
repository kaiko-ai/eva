"""Segmentation decoder heads API."""

from eva.vision.models.networks.decoders.segmentation.decoder2d import Decoder2D
from eva.vision.models.networks.decoders.segmentation.linear import LinearDecoder
from eva.vision.models.networks.decoders.segmentation.semantic import (
    ConvDecoder1x1,
    ConvDecoderMS,
    ConvDecoderWithImagePrior,
    SingleLinearDecoder,
)

__all__ = [
    "ConvDecoder1x1",
    "ConvDecoderMS",
    "SingleLinearDecoder",
    "ConvDecoderWithImagePrior",
    "Decoder2D",
    "LinearDecoder",
]
