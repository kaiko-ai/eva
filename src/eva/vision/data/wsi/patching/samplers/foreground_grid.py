"""Foreground grid sampler."""

from typing import Tuple

from eva.vision.data.wsi.patching.mask import Mask
from eva.vision.data.wsi.patching.samplers import _utils, base


class ForegroundGridSampler(base.ForegroundSampler):
    """Sample patches based on a grid, only returning patches containing foreground."""

    def __init__(
        self,
        max_samples: int = 20,
        overlap: Tuple[int, int] = (0, 0),
        min_foreground_ratio: float = 0.35,
        seed: int = 42,
    ) -> None:
        """Initializes the sampler.

        Args:
            max_samples: The maximum number of samples to return.
            overlap: The overlap between patches in the grid.
            min_foreground_ratio: The minimum amount of foreground
                within a sampled patch.
            seed: The random seed.
        """
        self.max_samples = max_samples
        self.overlap = overlap
        self.min_foreground_ratio = min_foreground_ratio
        self.seed = seed

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: Tuple[int, int],
        mask: Mask,
    ):
        """Sample patches from a grid containing foreground.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
            mask: The mask of the image.
        """
        _utils.validate_dimensions(width, height, layer_shape)
        x_y, indices = _utils.get_grid_coords_and_indices(
            layer_shape, width, height, self.overlap, seed=self.seed
        )

        count = 0
        for i in indices:
            if count >= self.max_samples:
                break

            if self.is_foreground(
                mask=mask,
                x=x_y[i][0],
                y=x_y[i][1],
                width=width,
                height=height,
                min_foreground_ratio=self.min_foreground_ratio,
            ):
                count += 1
                yield x_y[i]

    def is_foreground(
        self,
        mask: Mask,
        x: int,
        y: int,
        width: int,
        height: int,
        min_foreground_ratio: float,
    ) -> bool:
        """Check if a patch contains sufficient foreground.

        Args:
            mask: The mask of the image.
            x: The x-coordinate of the patch.
            y: The y-coordinate of the patch.
            width: The width of the patch.
            height: The height of the patch.
            min_foreground_ratio: The minimum amount of foreground in the patch.
        """
        x_, y_ = self._scale_coords(x, y, mask.scale_factors)
        width_, height_ = self._scale_coords(width, height, mask.scale_factors)
        patch_mask = mask.mask_array[y_ : y_ + height_, x_ : x_ + width_]
        return patch_mask.sum() / patch_mask.size >= min_foreground_ratio

    def _scale_coords(
        self,
        x: int,
        y: int,
        scale_factors: Tuple[float, float],
    ) -> Tuple[int, int]:
        return int(x / scale_factors[0]), int(y / scale_factors[1])
