"""Semantic Segmentation decoder heads API."""

from eva.vision.models.networks.decoders.segmentation.semantic.common import (
    ConvDecoder1x1,
    ConvDecoderMS,
    SingleLinearDecoder,
)
from eva.vision.models.networks.decoders.segmentation.semantic.swin_unetr import SwinUNETRDecoder
from eva.vision.models.networks.decoders.segmentation.semantic.with_image import (
    ConvDecoderWithImage,
)

__all__ = [
    "ConvDecoder1x1",
    "ConvDecoderMS",
    "ConvDecoderWithImage",
    "SingleLinearDecoder",
    "SwinUNETRDecoder",
]
