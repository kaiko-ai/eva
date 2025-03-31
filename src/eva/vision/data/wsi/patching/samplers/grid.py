"""Grid sampler."""

from typing import Generator, Tuple

from eva.vision.data.wsi.patching.samplers import _utils, base


class GridSampler(base.Sampler):
    """Sample patches based on a grid."""

    def __init__(
        self,
        max_samples: int | None = None,
        overlap: Tuple[int, int] = (0, 0),
        seed: int = 42,
        validate_dimensions: bool = True,
        include_last: bool = False,
    ):
        """Initializes the sampler.

        Args:
            max_samples: The maximum number of samples to return.
            overlap: The overlap between patches in the grid.
            seed: The random seed.
            validate_dimensions: Whether to validate the dimensions the image. It
                expects the patch size to be smaller than the image size.
            include_last: Whether to include coordinates of the last patch when it
                it partially exceeds the image.
        """
        self.max_samples = max_samples
        self.overlap = overlap
        self.seed = seed
        self.validate_dimensions = validate_dimensions
        self.include_last = include_last

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
        if self.validate_dimensions:
            _utils.validate_dimensions(width, height, layer_shape)
        x_y, indices = _utils.get_grid_coords_and_indices(
            layer_shape, width, height, self.overlap, seed=self.seed, include_last=self.include_last
        )
        max_samples = len(indices) if self.max_samples is None else self.max_samples
        for i in indices[:max_samples]:
            yield x_y[i]
