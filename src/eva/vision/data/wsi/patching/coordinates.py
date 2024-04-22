"""A module for handling coordinates of patches from a whole-slide image."""

import dataclasses
from typing import List, Tuple

from eva.vision.data import wsi
from eva.vision.data.wsi.patching import samplers


@dataclasses.dataclass
class PatchCoordinates:
    """A class to store coordinates of patches from a whole-slide image.

    Args:
        x_y: A list of (x, y) coordinates of the patches.
        width: The width of the patches, in pixels.
        height: The height of the patches, in pixels.
        level_idx: The level index of the patches.
    """

    x_y: List[Tuple[int, int]]
    width: int
    height: int
    level_idx: int

    @classmethod
    def from_file(
        cls,
        wsi_path: str,
        width: int,
        height: int,
        target_mpp: float,
        sampler: samplers.Sampler,
        backend: wsi.WsiBackend = wsi.WsiBackend.OPENSLIDE,
    ) -> "PatchCoordinates":
        """Create a new instance of PatchCoordinates from a whole-slide image file.

        Patches will be read from the level that is closest to the specified target_mpp.

        Args:
            wsi_path: The path to the whole-slide image file.
            width: The width of the patches to be extracted, in pixels.
            height: The height of the patches to be extracted, in pixels.
            target_mpp: The target microns per pixel (mpp) for the patches.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
        """
        wsi_obj = wsi.get_wsi_class(backend)(wsi_path)
        wsi_obj.open_slide()
        x_y = []
        level_idx = wsi_obj.get_closest_level(target_mpp)
        level_mpp = wsi_obj.mpp * wsi_obj.level_downsamples[level_idx]
        mpp_ratio = target_mpp / level_mpp
        scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)

        for x, y in sampler.sample(
            scaled_width, scaled_height, wsi_obj.level_dimensions[level_idx]
        ):
            x_y.append((x, y))

        return cls(x_y, scaled_width, scaled_height, level_idx)
