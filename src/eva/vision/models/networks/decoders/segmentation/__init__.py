"""Segmentation decoder heads API."""

from eva.vision.models.networks.decoders.segmentation.common import (
    ConvDecoder1x1,
    ConvDecoderMS,
    SingleLinearDecoder,
)
from eva.vision.models.networks.decoders.segmentation.conv2d import ConvDecoder
from eva.vision.models.networks.decoders.segmentation.linear import LinearDecoder

__all__ = ["ConvDecoder1x1", "ConvDecoderMS", "SingleLinearDecoder", "ConvDecoder", "LinearDecoder"]
