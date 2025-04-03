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
        include_partial_patches: bool = False,
    ):
        """Initializes the sampler.

        Args:
            max_samples: The maximum number of samples to return.
            overlap: The overlap between patches in the grid.
            seed: The random seed.
            include_partial_patches: Whether to include coordinates of the last patch when it
                it partially exceeds the image and therefore is smaller than the
                specified patch size.
        """
        self.max_samples = max_samples
        self.overlap = overlap
        self.seed = seed
        self.include_partial_patches = include_partial_patches

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
        x_y = _utils.get_grid_coords(
            layer_shape,
            width,
            height,
            self.overlap,
            seed=self.seed,
            include_partial_patches=self.include_partial_patches,
        )
        max_samples = len(x_y) if self.max_samples is None else min(self.max_samples, len(x_y))
        for i in range(max_samples):
            yield x_y[i]
