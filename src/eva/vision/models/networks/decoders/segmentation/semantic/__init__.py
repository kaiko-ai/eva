"""Semantic Segmentation decoder heads API."""

from eva.vision.models.networks.decoders.segmentation.semantic.common import (
    ConvDecoder1x1,
    ConvDecoderMS,
    SingleLinearDecoder,
)
from eva.vision.models.networks.decoders.segmentation.semantic.with_image_prior import (
    ConvDecoderWithImagePrior,
)

__all__ = ["ConvDecoder1x1", "ConvDecoderMS", "SingleLinearDecoder", "ConvDecoderWithImagePrior"]
