"""Image encoding utilities."""

import base64
import io
import os
from typing import Literal

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def encode_image(
    image: tv_tensors.Image,
    encoding: Literal["base64"] = "base64",
    file_format: Literal["png", "jpeg"] = "jpeg",
    **kwargs,
) -> str:
    """Encodes an image tensor into a string format.

    Args:
        image: The image tensor to encode.
        encoding: The encoding format to use. Currently only supports "base64".
        file_format: The file format to encode the image as. Either "png" or "jpeg".
            Can be overridden by the ENCODE_IMAGE_FORMAT environment variable.
        **kwargs: Additional keyword arguments to pass to the encoding function.

    Returns:
        An encoded string representation of the image.
    """
    image_format = os.getenv("ENCODE_IMAGE_FORMAT", file_format).lower()

    if image_format not in {"png", "jpeg"}:
        raise ValueError("Unsupported format: use 'png' or 'jpeg'.")

    match encoding:
        case "base64":
            return _encode_base64(image, file_format=image_format, **kwargs)  # type: ignore
        case _:
            raise ValueError(f"Unsupported encoding type: {encoding}. Supported: 'base64'")


def _encode_base64(
    image: tv_tensors.Image,
    file_format: Literal["png", "jpeg"] = "jpeg",
    optimize: bool = False,
    compress_level: int = 4,
    quality: int = 95,
) -> str:
    """Encodes an image tensor as a base64 string in PNG or JPEG format.

    Args:
        image: Image tensor to encode.
        file_format: Either "png" (lossless) or "jpeg" (lossy).
        optimize: If True, performs an extra optimization pass for slightly smaller files
            at the cost of slower encoding.
        compress_level: PNG-only. Compression level (0-9); lower is faster, but results
            in slightly bigger files.
        quality: JPEG-only. Quality (1-100); higher gives better visual quality.
            Defaults to 95, which is visually lossless for most images.

    Returns:
        Base64-encoded string of the image in the chosen format.
    """
    buf = _get_image_buffer(
        image,
        file_format=file_format,
        optimize=optimize,
        compress_level=compress_level,
        quality=quality,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_image_buffer(
    image: tv_tensors.Image,
    file_format: Literal["png", "jpeg"] = "jpeg",
    optimize: bool = False,
    compress_level: int = 4,
    quality: int = 95,
) -> io.BytesIO:
    """Converts an image tensor into an in-memory file buffer in PNG or JPEG format."""
    buf = io.BytesIO()
    pil_img = F.to_pil_image(image)

    match file_format:
        case "jpeg":
            pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=optimize)
        case "png":
            pil_img.save(buf, format="PNG", compress_level=compress_level, optimize=optimize)
        case _:
            raise ValueError("Unsupported format: use 'png' or 'jpeg'.")

    buf.seek(0)
    return buf
