"""Image conversion related functionalities."""

from typing import Iterable

import torch
from torchvision.transforms.v2 import functional


def descale_and_denorm_image(
    image: torch.Tensor,
    mean: Iterable[float] = (0.0, 0.0, 0.0),
    std: Iterable[float] = (1.0, 1.0, 1.0),
    inplace: bool = True,
) -> torch.Tensor:
    """De-scales and de-norms an image tensor to (0, 255) range.

    Args:
        image: An image float tensor.
        mean: The mean that the image channels are normalized with.
        std: The std that the image channels are normalized with.
        inplace: Whether to perform the operation in-place.

    Returns:
        The image tensor of range (0, 255) range as uint8.
    """
    if not inplace:
        image = image.clone()

    norm_image = _descale_image(image, mean=mean, std=std)
    return _denorm_image(norm_image)


def _descale_image(
    image: torch.Tensor,
    mean: Iterable[float] = (0.0, 0.0, 0.0),
    std: Iterable[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """De-scales an image tensor to (0., 1.) range.

    Args:
        image: An image float tensor.
        mean: The normalized channels mean values.
        std: The normalized channels std values.

    Returns:
        The de-normalized image tensor of range (0., 1.).
    """
    return functional.normalize(
        image,
        mean=[-cmean / cstd for cmean, cstd in zip(mean, std, strict=False)],
        std=[1 / cstd for cstd in std],
    )


def _denorm_image(image: torch.Tensor) -> torch.Tensor:
    """De-normalizes an image tensor from (0., 1.) to (0, 255) range.

    Args:
        image: An image float tensor.

    Returns:
        The image tensor of range (0, 255) range as uint8.
    """
    image_scaled = image - image.min()
    image_scaled /= image_scaled.max()
    image_scaled *= 255
    return image_scaled.to(dtype=torch.uint8)
