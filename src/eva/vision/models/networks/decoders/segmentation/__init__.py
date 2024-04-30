"""Segmentation decoder heads API."""

from eva.vision.models.networks.decoders.segmentation.conv import ConvDecoder
from eva.vision.models.networks.decoders.segmentation.linear import LinearDecoder

__all__ = ["ConvDecoder", "LinearDecoder"]
