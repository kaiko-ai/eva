"""Functional resizing utilities."""

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from eva.vision.utils.image import encode as encode_utils


def resize_to_max_bytes(image: tv_tensors.Image, max_bytes: int) -> tv_tensors.Image:
    """Resize the image to fit within the specified byte size.

    Args:
        image: The image tensor to resize.
        max_bytes: The maximum allowed byte size for the image.

    Returns:
        The resized image tensor.
    """
    num_bytes = len(encode_utils.encode_image(image))
    h, w = image.shape[-2], image.shape[-1]

    while num_bytes > max_bytes:
        scale = (max_bytes / num_bytes) ** 0.5
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        if new_h == h and new_w == w:
            break

        image = tv_tensors.Image(F.resize(image, [new_h, new_w]))
        h, w = new_h, new_w
        num_bytes = len(encode_utils.encode_image(image))

    return image
