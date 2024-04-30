"""A module for handling coordinates of patches from a whole-slide image."""

import dataclasses
import functools
from typing import List, Tuple

import cv2
import numpy as np

from eva.vision.data.wsi import backends
from eva.vision.data.wsi.backends.base import Wsi
from eva.vision.data.wsi.patching import samplers

LRU_CACHE_SIZE = 32


@dataclasses.dataclass
class PatchCoordinates:
    """A class to store coordinates of patches from a whole-slide image.

    Args:
        x_y: A list of (x, y) coordinates of the patches.
        width: The width of the patches, in pixels (refers to x-dim).
        height: The height of the patches, in pixels (refers to y-dim).
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
        backend: str = "openslide",
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
        wsi = backends.wsi_backend(backend)(wsi_path)
        level_idx = wsi.get_closest_level(target_mpp)
        level_mpp = wsi.mpp * wsi.level_downsamples[level_idx]
        mpp_ratio = target_mpp / level_mpp
        scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)

        sample_args = {
            "width": scaled_width,
            "height": scaled_height,
            "layer_shape": wsi.level_dimensions[level_idx],
        }
        if isinstance(sampler, samplers.ForegroundSampler):
            sample_args["mask"] = get_mask(wsi, level_idx)

        x_y = [(x, y) for x, y in sampler.sample(**sample_args)]

        return cls(x_y, scaled_width, scaled_height, level_idx)


@functools.lru_cache(LRU_CACHE_SIZE)
def get_cached_coords(
    file_path: str,
    width: int,
    height: int,
    target_mpp: float,
    sampler: samplers.Sampler,
    backend: str,
) -> PatchCoordinates:
    """Get a cached instance of PatchCoordinates for the specified parameters."""
    return PatchCoordinates.from_file(
        wsi_path=file_path,
        width=width,
        height=height,
        target_mpp=target_mpp,
        backend=backend,
        sampler=sampler,
    )


def get_mask(
    wsi: Wsi,
    level_idx: int,
    kernel_size: tuple[int, int] = (7, 7),
    gray_threshold: int = 220,
    fill_holes: bool = False,
) -> tuple[np.ndarray, float]:
    """Extracts a binary mask from an image.

    Args:
        wsi: The WSI object.
        level_idx: The level index to extract the mask from.
        kernel_size: The size of the kernel for morphological operations.
        gray_threshold: The threshold for the gray scale image.
        fill_holes: Whether to fill holes in the mask.
    """
    image = np.array(
        wsi.read_region([0, 0], len(wsi.level_dimensions) - 1, wsi.level_dimensions[-1])
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)

    if fill_holes:
        mask = cv2.dilate(mask, kernel, iterations=1)
        contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask, [cnt], 0, 1, -1)

    mask_scale_factor = wsi.level_dimensions[-1][0] / wsi.level_dimensions[level_idx][0]

    return mask, mask_scale_factor
