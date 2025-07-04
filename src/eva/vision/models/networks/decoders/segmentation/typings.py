"""Type-hints for segmentation decoders."""

from typing import List, NamedTuple, Tuple

import torch


class DecoderInputs(NamedTuple):
    """Input scheme for segmentation decoders."""

    features: List[torch.Tensor]
    """List of image features generated by the encoder from the original images."""

    image_size: Tuple[int, ...]
    """Size of the original input images to be used for upsampling."""

    images: torch.Tensor | None = None
    """The original input images for which the encoder generated the encoded_images."""
