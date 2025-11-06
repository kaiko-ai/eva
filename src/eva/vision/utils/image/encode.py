"""Image encoding utilities."""

import base64
import io
import os
from typing import Literal

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def encode_image(image: tv_tensors.Image, encoding: Literal["base64"], **kwargs) -> str:
    """Encodes an image tensor into a string format.

    Args:
        image: The image tensor to encode.
        encoding: The encoding format to use. Currently only supports "base64".
        **kwargs: Additional keyword arguments to pass to the encoding function.

    Returns:
        An encoded string representation of the image.
    """
    match encoding:
        case "base64":
            return encode_base64(image, **kwargs)
        case _:
            raise ValueError(f"Unsupported encoding type: {encoding}. Supported: 'base64'")


def encode_base64(
    image: tv_tensors.Image,
    file_format: Literal["png", "jpeg"] = "jpeg",
    optimize: bool = False,
    compress_level: int = 4,
    quality: int = 95,
) -> str:
    """Encodes an image tensor as a base64 string in PNG or JPEG format.

    Args:
        image: Image tensor to encode.
        file_format: Either "png" (lossless) or "jpeg" (lossy). Can be overridden
            by the BASE64_IMAGE_FORMAT environment variable.
        optimize: If True, performs an extra optimization pass for slightly smaller files
            at the cost of slower encoding.
        compress_level: PNG-only. Compression level (0-9); lower is faster, but results
            in slightly bigger files.
        quality: JPEG-only. Quality (1-100); higher gives better visual quality.
            Defaults to 95, which is visually lossless for most images.

    Returns:
        Base64-encoded string of the image in the chosen format.
    """
    buf = get_image_buffer(
        image,
        file_format=file_format,
        optimize=optimize,
        compress_level=compress_level,
        quality=quality,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_image_buffer(
    image: tv_tensors.Image,
    file_format: Literal["png", "jpeg"] = "jpeg",
    optimize: bool = False,
    compress_level: int = 4,
    quality: int = 95,
) -> io.BytesIO:
    """Converts an image tensor into an in-memory file buffer in PNG or JPEG format."""
    buf = io.BytesIO()
    pil_img = F.to_pil_image(image)

    match os.getenv("BASE64_IMAGE_FORMAT", file_format).lower():
        case "jpeg":
            pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=optimize)
        case "png":
            pil_img.save(buf, format="PNG", compress_level=compress_level, optimize=optimize)
        case _:
            raise ValueError("Unsupported format: use 'png' or 'jpeg'.")

    buf.seek(0)
    return buf
