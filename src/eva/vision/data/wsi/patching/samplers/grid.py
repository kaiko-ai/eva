"""Grid sampler."""

from typing import Generator, Tuple

from eva.vision.data.wsi.patching.samplers import _utils, base


class GridSampler(base.Sampler):
    """Sample patches based on a grid.

    Note: If `max_samples` is reached before the entire grid is traversed, sampling
    stops early and only the collected patches are returned.

    Args:
        max_samples: The maximum number of samples to return.
        overlap: The overlap between patches in the grid.
        shuffle: Whether to shuffle the grid indices before sampling.
            If True, patches are randomly distributed across the WSI.
            If False, patches are sampled in column-major order from top-left.
        seed: The random seed.
    """

    def __init__(
        self,
        max_samples: int | None = None,
        overlap: Tuple[int, int] = (0, 0),
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initializes the sampler."""
        self.max_samples = max_samples
        self.overlap = overlap
        self.shuffle = shuffle
        self.seed = seed

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: Tuple[int, int],
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample patches from a grid.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
        """
        _utils.validate_dimensions(width, height, layer_shape)
        x_y, indices = _utils.get_grid_coords_and_indices(
            layer_shape, width, height, self.overlap, shuffle=self.shuffle, seed=self.seed
        )
        max_samples = len(indices) if self.max_samples is None else self.max_samples
        for i in indices[:max_samples]:
            yield x_y[i]
