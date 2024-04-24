"""Decoder heads API."""

from eva.vision.models.networks.decoders.conv import ConvDecoder
from eva.vision.models.networks.decoders.decoder import Decoder
from eva.vision.models.networks.decoders.linear import LinearDecoder

__all__ = ["ConvDecoder", "Decoder", "LinearDecoder"]
