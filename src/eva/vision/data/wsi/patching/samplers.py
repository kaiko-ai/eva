"""Samplers for WSI patch extraction."""

import abc
import random
from typing import Generator, Tuple

import numpy as np


class Sampler(abc.ABC):
    """Base class for samplers."""

    @abc.abstractmethod
    def sample(
        self,
        width: int,
        height: int,
        layer_shape: tuple[int, int],
        *args,
    ) -> Generator[Tuple[int, int], None, None]:
        """Iterator that samples patches."""


class ForegroundSampler(Sampler):
    """Base class for samplers with foreground filtering capabilities."""

    @abc.abstractmethod
    def sample(
        self,
        width: int,
        height: int,
        layer_shape: tuple[int, int],
        mask: tuple[np.ndarray, float],
        *args,
    ) -> Generator[Tuple[int, int], None, None]:
        """Iterator that samples patches."""

    @abc.abstractmethod
    def is_foreground(
        self,
        mask: tuple[np.ndarray, float],
        x: int,
        y: int,
        width: int,
        height: int,
        min_patch_info=0.35,
    ) -> bool:
        """Check if a patch contains sufficient foreground."""

class RandomSampler(Sampler):
    """Sample patch coordinates randomly.

    Args:
        n_samples: The number of samples to return.
        seed: The random seed.
    """

    def __init__(self, n_samples: int = 1, seed: int = 42):
        """Initializes the sampler."""
        self.seed = seed
        self.n_samples = n_samples

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: tuple[int, int],
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample random patches.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
        """
        _set_seed(self.seed)

        for _ in range(self.n_samples):
            x_max, y_max = layer_shape[0], layer_shape[1]
            x, y = random.randint(0, x_max - width), random.randint(0, y_max - height)  # nosec
            yield x, y


class GridSampler(Sampler):
    """Sample patches based on a grid.

    Args:
        max_samples: The maximum number of samples to return.
        overlap: The overlap between patches in the grid.
        seed: The random seed.
    """

    def __init__(
        self,
        max_samples: int | None = None,
        overlap: tuple[int, int] = (0, 0),
        seed: int = 42,
    ):
        """Initializes the sampler."""
        self.max_samples = max_samples
        self.overlap = overlap
        self.seed = seed

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: tuple[int, int],
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample patches from a grid.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
        """

        x_y, indices = _get_grid_coords_and_indices(
            layer_shape,
            width,
            height,
            self.overlap
        )
        max_samples = len(indices) if self.max_samples is None else self.max_samples
        for i in indices[:max_samples]:
            yield x_y[i]


class ForegroundGridSampler(ForegroundSampler):
    """Sample patches based on a grid, only returning patches containing foreground.
    
    Args:
        max_samples: The maximum number of samples to return.
    """
    def __init__(
        self,
        max_samples: int = 20,
        overlap: tuple[int, int] = (0, 0),
        seed: int = 42,
    ):
        """Initializes the sampler."""
        self.max_samples = max_samples
        self.overlap = overlap
        self.seed = seed

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: tuple[int, int],
        mask: tuple[np.ndarray, float],
    ):
        """Sample patches from a grid containing foreground.
        
        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
            mask: The mask of the image.
            mask_scale_factor: The scale factor of the mask.
        """
        x_y, indices = _get_grid_coords_and_indices(
            layer_shape,
            width,
            height,
            self.overlap
        )

        count = 0
        for i in indices:
            if count >= self.max_samples:
                break
            if self.is_foreground(
                mask, x_y[i][0], x_y[i][1], width, height
            ):
                count += 1
                yield x_y[i]


    def is_foreground(
        self,
        mask: tuple[np.ndarray, float],
        x: int,
        y: int,
        width: int,
        height: int,
        min_patch_info=0.35,
    ) -> bool:
        """Check if a patch contains sufficient foreground.
        
        Args:
            mask: The mask of the image.
            x: The x-coordinate of the patch.
            y: The y-coordinate of the patch.
            width: The width of the patch.
            height: The height of the patch.
            mask_scale_factor: The scale factor of the mask.
            min_patch_info: The minimum amount of foreground in the patch.
        """
        mask_array, mask_scale_factor = mask
        x_, y_, width_, height_ = self.scale_coords(mask_scale_factor, x, y, width, height)
        patch_mask = mask_array[y_ : y_ + height_, x_ : x_ + width_]
        return patch_mask.sum() / patch_mask.size > min_patch_info

    def scale_coords(self, scale_factor, *coords):
        return tuple(int(coord * scale_factor) for coord in coords)


def _get_grid_coords_and_indices(
    layer_shape: tuple[int, int],
    width: int,
    height: int,
    overlap: tuple[int, int],
    shuffle: bool = True,
    seed: int = 42,
):
    """Get grid coordinates and indices.
    
    Args:
        layer_shape: The shape of the layer.
        width: The width of the patches.
        height: The height of the patches.
        overlap: The overlap between patches in the grid.
        shuffle: Whether to shuffle the indices.
        seed: The random seed.
    """
    x_range = range(0, layer_shape[0] - width, width - overlap[0])
    y_range = range(0, layer_shape[1] - height, height - overlap[1])
    x_y = [(x, y) for x in x_range for y in y_range]

    indices = list(range(len(x_y)))
    if shuffle:
        _set_seed(seed)
        np.random.shuffle(indices)
    return x_y, indices


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
