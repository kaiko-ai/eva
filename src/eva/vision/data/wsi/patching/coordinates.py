"""A module for handling coordinates of patches from a whole-slide image."""

import dataclasses
import functools
from typing import Any, Dict, List, Tuple

from eva.vision.data.wsi import backends
from eva.vision.data.wsi.patching import samplers
from eva.vision.data.wsi.patching.mask import Mask, get_mask, get_mask_level

LRU_CACHE_SIZE = 32


@dataclasses.dataclass
class PatchCoordinates:
    """A class to store coordinates of patches from a whole-slide image.

    Args:
        x_y: A list of (x, y) coordinates of the patches (refer to level 0).
        width: The width of the patches, in pixels (refers to level_idx).
        height: The height of the patches, in pixels (refers to level_idx).
        level_idx: The level index at which to extract the patches.
        mask: The foreground mask of the wsi.
    """

    x_y: List[Tuple[int, int]]
    width: int
    height: int
    level_idx: int
    mask: Mask | None = None

    @classmethod
    def from_file(
        cls,
        wsi_path: str,
        width: int,
        height: int,
        sampler: samplers.Sampler,
        target_mpp: float,
        overwrite_mpp: float | None = None,
        backend: str = "openslide",
    ) -> "PatchCoordinates":
        """Create a new instance of PatchCoordinates from a whole-slide image file.

        Patches will be read from the level that is closest to the specified target_mpp.

        Args:
            wsi_path: The path to the whole-slide image file.
            width: The width of the patches to be extracted, in pixels.
            height: The height of the patches to be extracted, in pixels.
            target_mpp: The target microns per pixel (mpp) for the patches.
            overwrite_mpp: The microns per pixel (mpp) value to use when missing in WSI metadata.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
        """
        wsi = backends.wsi_backend(backend)(wsi_path, overwrite_mpp)

        # Sample patch coordinates at level 0
        mpp_ratio_0 = target_mpp / wsi.mpp
        sample_args = {
            "width": int(mpp_ratio_0 * width),
            "height": int(mpp_ratio_0 * height),
            "layer_shape": wsi.level_dimensions[0],
        }
        if isinstance(sampler, samplers.ForegroundSampler):
            mask_level_idx = get_mask_level(wsi, width, height, target_mpp)
            sample_args["mask"] = get_mask(wsi, mask_level_idx)

        x_y = list(sampler.sample(**sample_args))

        # Scale dimensions to level that is closest to the target_mpp
        level_idx = wsi.get_closest_level(target_mpp)
        mpp_ratio = target_mpp / (wsi.mpp * wsi.level_downsamples[level_idx])
        scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)

        return cls(x_y, scaled_width, scaled_height, level_idx, sample_args.get("mask"))

    def to_dict(self, include_keys: List[str] | None = None) -> Dict[str, Any]:
        """Convert the coordinates to a dictionary."""
        include_keys = include_keys or ["x_y", "width", "height", "level_idx"]
        coord_dict = dataclasses.asdict(self)
        if include_keys:
            coord_dict = {key: coord_dict[key] for key in include_keys}
        return coord_dict


@functools.lru_cache(LRU_CACHE_SIZE)
def get_cached_coords(
    file_path: str,
    width: int,
    height: int,
    target_mpp: float,
    overwrite_mpp: float | None,
    sampler: samplers.Sampler,
    backend: str,
) -> PatchCoordinates:
    """Get a cached instance of PatchCoordinates for the specified parameters."""
    return PatchCoordinates.from_file(
        wsi_path=file_path,
        width=width,
        height=height,
        target_mpp=target_mpp,
        overwrite_mpp=overwrite_mpp,
        backend=backend,
        sampler=sampler,
    )
