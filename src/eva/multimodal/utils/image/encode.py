"""Image encoding utilities."""

import base64
import io
from typing import Literal

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def encode_image(image: tv_tensors.Image, encoding: Literal["base64"]) -> str:
    """Encodes an image tensor into a string format.

    Args:
        image: The image tensor to encode.
        encoding: The encoding format to use. Currently only supports "base64".

    Returns:
        An encoded string representation of the image.
    """
    match encoding:
        case "base64":
            image_bytes = io.BytesIO()
            F.to_pil_image(image).save(image_bytes, format="PNG", optimize=True)
            image_bytes.seek(0)
            return base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        case _:
            raise ValueError(f"Unsupported encoding type: {encoding}. Supported: 'base64'")
