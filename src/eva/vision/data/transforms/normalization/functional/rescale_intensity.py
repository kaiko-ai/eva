"""Intensity level functions."""

import sys
from typing import Tuple

import torch


def rescale_intensity(
    image: torch.Tensor,
    in_range: Tuple[float, float] | None = None,
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Stretches or shrinks the image intensity levels.

    Args:
        image: The image tensor as float-type.
        in_range: The input data range. If `None`, it will
            fetch the min and max of the input image.
        out_range: The desired intensity range of the output.

    Returns:
        The image tensor after stretching or shrinking its intensity levels.
    """
    imin, imax = in_range or (image.min(), image.max())
    omin, omax = out_range
    image_scaled = (image - imin) / (imax - imin + sys.float_info.epsilon)
    return image_scaled * (omax - omin) + omin
