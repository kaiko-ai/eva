"""Base classes for samplers."""

import abc
from typing import Generator, Tuple

from eva.vision.data.wsi.patching.mask import Mask


class Sampler(abc.ABC):
    """Base class for samplers."""

    @abc.abstractmethod
    def sample(
        self,
        width: int,
        height: int,
        layer_shape: Tuple[int, int],
        mask: Mask | None = None,
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample patche coordinates.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
            mask: Tuple containing the mask array and the scaling factor with respect to the
                provided layer_shape. Optional, only required for samplers with foreground
                filtering.

        Returns:
            A generator producing sampled patch coordinates.
        """


class ForegroundSampler(Sampler):
    """Base class for samplers with foreground filtering capabilities."""

    @abc.abstractmethod
    def is_foreground(
        self,
        mask: Mask,
        x: int,
        y: int,
        width: int,
        height: int,
        min_foreground_ratio: float,
    ) -> bool:
        """Check if a patch contains sufficient foreground."""
