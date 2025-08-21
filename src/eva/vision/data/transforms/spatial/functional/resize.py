"""Functional resizing utilities."""

import io
from typing import Tuple

from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def resize_to_max_bytes(image: tv_tensors.Image, max_bytes: int) -> tv_tensors.Image:
    """Resize the image to fit within the specified byte size."""
    image_pil = F.to_pil_image(image)
    image_bytes = io.BytesIO()
    image_pil.save(image_bytes, format="PNG", optimize=True)

    while image_bytes.tell() > max_bytes:
        size: Tuple[int, int] = image_pil.size  # type: ignore
        w, h = size
        scale = (max_bytes / image_bytes.tell()) ** 0.5
        new_size = (max(1, int(h * scale)), max(1, int(w * scale)))
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="PNG", optimize=True)

    return tv_tensors.Image(F.pil_to_tensor(image_pil))
