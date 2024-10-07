"""Random sampler."""

import random
from typing import Generator, Tuple

from eva.vision.data.wsi.patching.samplers import _utils, base


class RandomSampler(base.Sampler):
    """Sample patch coordinates randomly.

    Args:
        n_samples: The number of samples to return.
        seed: The random seed.
    """

    def __init__(self, n_samples: int = 1, seed: int = 42):
        """Initializes the sampler."""
        self.seed = seed
        self.n_samples = n_samples
        self.random_generator = random.Random(seed)  # nosec

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: Tuple[int, int],
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample random patches.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
        """
        _utils.validate_dimensions(width, height, layer_shape)

        x_max, y_max = layer_shape[0], layer_shape[1]
        for _ in range(self.n_samples):
            x, y = self.random_generator.randint(0, x_max - width), self.random_generator.randint(
                0, y_max - height
            )
            yield x, y
