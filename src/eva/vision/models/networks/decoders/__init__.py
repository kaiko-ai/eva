"""Decoder networks API."""

from eva.vision.models.networks.decoders.convolutional import ConvDecoder
from eva.vision.models.networks.decoders.linear import LinearDecoder

__all__ = ["ConvDecoder", "LinearDecoder"]
