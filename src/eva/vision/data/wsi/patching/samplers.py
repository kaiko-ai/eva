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
    ) -> Generator[Tuple[int, int], None, None]:
        """Iterator that samples patches."""


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
            x_max, y_max = layer_shape[1], layer_shape[0]  # TODO: check if this is correct
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
        _set_seed(self.seed)

        x_range = range(0, layer_shape[1], width - self.overlap[0])
        y_range = range(0, layer_shape[0], height - self.overlap[1])
        x_y = [(x, y) for x in x_range for y in y_range]

        shuffled_indices = (
            np.random.choice(len(x_y), self.max_samples, replace=False)
            if self.max_samples
            else range(len(x_y))
        )
        for i in shuffled_indices:
            yield x_y[i]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
