"""Samplers for WSI patch extraction."""

import abc
import math
import random
from typing import Generator, Tuple

import cv2
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
        ignore_max_samples: bool = False,
    ) -> Generator[Tuple[int, int], None, None]:
        """Sample patches from a grid.

        Args:
            width: The width of the patches.
            height: The height of the patches.
            layer_shape: The shape of the layer.
        """
        _set_seed(self.seed)

        x_range = range(0, layer_shape[0] - width, width - self.overlap[0])
        y_range = range(0, layer_shape[1] - height, height - self.overlap[1])
        x_y = [(x, y) for x in x_range for y in y_range]

        indices = list(range(len(x_y)))
        np.random.shuffle(indices)
                       
        shuffled_indices = (
            np.random.choice(len(x_y), self.max_samples, replace=False)
            if self.max_samples
            else range(len(x_y))
        )
        if self.max_samples is not None and not ignore_max_samples:
            for i in indices[:self.max_samples]:
                yield x_y[i]
        else:
            for i_, i in enumerate(indices):
                yield x_y[i]


class ForegroundGridSampler(GridSampler):
    def __init__(
        self, 
        max_samples: int=20,
    ):
        super().__init__(max_samples=max_samples)

    def sample(
        self,
        width: int,
        height: int,
        layer_shape: tuple[int, int],
        image: np.ndarray,
        # patch_size_scaled: int,
        scale_factor: float,
    ):
        count = 0
        for x, y in super().sample(width, height, layer_shape, ignore_max_samples=True):
            if count >= self.max_samples:
                break
            if self.is_foreground(image, x, y, width, height, scale_factor):
                count += 1
                # print(x,y)
                yield x, y

    def is_foreground(
        self, 
        image: np.ndarray, 
        x: int, 
        y: int, 
        width: int, 
        height: int,
        scale_factor: float,
        min_patch_info=0.35,
        # min_patch_info=0.65,
    ) -> bool:
        mask = self.get_mask(image)
        x_, y_, width_, height_ = self.scale_coords(scale_factor, x, y, width, height)
        # patch_mask = mask[x_:x_ + width_, y_:y_ + height_]
        patch_mask = mask[ y_:y_ + height_, x_:x_ + width_]
        return patch_mask.sum() / patch_mask.size > min_patch_info


    def scale_coords(self, scale_factor, *coords):
        return tuple(int(coord * scale_factor) for coord in coords)


    def get_mask(self, image, kernel_size=(7, 7), gray_threshold=220):
        # Define elliptic kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        # Convert rgb to gray scale for easier masking
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Now mask the gray-scaled image (capturing tissue in biopsy)
        mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
        # Use dilation and findContours to fill in gaps/holes in masked tissue
        mask = cv2.dilate(mask, kernel, iterations=1)
        # contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contour:
        #     cv2.drawContours(mask, [cnt], 0, 1, -1)
        return mask



# def mask_tissue(self, image, kernel_size=(7, 7), gray_threshold=220):
#     """Masks tissue in image. Uses gray-scaled image, as well as
#     dilation kernels and 'gap filling'
#     """
#     # Define elliptic kernel
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
#     # Convert rgb to gray scale for easier masking
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # Now mask the gray-scaled image (capturing tissue in biopsy)
#     mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
#     # Use dilation and findContours to fill in gaps/holes in masked tissue
#     mask = cv2.dilate(mask, kernel, iterations=1)
#     contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contour:
#         cv2.drawContours(mask, [cnt], 0, 1, -1)
#     return mask



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# class RandomForegroundSampler(Sampler):
#     def __init__(
#         self,
#         max_n_samples: int = 1,
#         seed: int = 42,
#         min_patch_info=0.35,
#         min_axis_info=0.35,
#         min_consec_axis_info=0.35,
#         min_decimal_keep=0.7,
#     ):
#         """Initializes the sampler."""
#         self.seed = seed
#         self.max_n_samples = max_n_samples
#         self.min_patch_info = min_patch_info
#         self.min_axis_info = min_axis_info
#         self.min_consec_axis_info = min_consec_axis_info
#         self.min_decimal_keep = min_decimal_keep

#     def sample(
#         self,
#         width: int,
#         height: int,
#         layer_shape: tuple[int, int],
#         image: np.ndarray,
#         patch_size_scaled: int,
#         downscale_factor: float,
#     ) -> Generator[Tuple[int, int], None, None]:
#         """Sample random patches.

#         Args:
#             width: The width of the patches.
#             height: The height of the patches.
#             layer_shape: The shape of the layer.
#         """
#         _set_seed(self.seed)

#         # masked tissue will be used to compute the coordinates
#         mask = self.mask_tissue(image)

#         # initialize coordinate accumulator
#         coords = np.zeros([0, 2], dtype=int)

#         y_sum = mask.sum(axis=1)
#         x_sum = mask.sum(axis=0)
#         # if on bits in x_sum is greater than in y_sum, the tissue is
#         # likely aligned horizontally. The algorithm works better if
#         # the image is aligned vertically, thus the image will be transposed
#         if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
#             image = self.transpose_image(image)
#             mask = self.transpose_image(mask)
#             y_sum, _ = x_sum, y_sum

#         # where y_sum is more than the minimum number of on-bits
#         y_tissue = np.where(y_sum >= (patch_size_scaled * self.min_axis_info))[0]

#         if len(y_tissue) < 1:
#             print("Not enough tissue in image (y-dim)", RuntimeWarning)
#             return [(0, 0, 0)]

#         y_tissue_parts_indices = self.get_tissue_parts_indices(
#             y_tissue, patch_size_scaled * self.min_consec_axis_info
#         )

#         if len(y_tissue_parts_indices) < 1:
#             print("Not enough tissue in image (y-dim)", RuntimeWarning)
#             return [(0, 0, 0)]

#         # loop over the tissues in y-dimension
#         for yidx in y_tissue_parts_indices:
#             y_tissue_subparts_coords = self.get_tissue_subparts_coords(
#                 yidx, patch_size_scaled, self.min_decimal_keep
#             )

#             for y in y_tissue_subparts_coords:
#                 # in y_slice, where x_slice_sum is more than the minimum number of on-bits
#                 x_slice_sum = mask[y : y + patch_size_scaled, :].sum(axis=0)
#                 x_tissue = np.where(x_slice_sum >= (patch_size_scaled * self.min_axis_info))[0]

#                 x_tissue_parts_indices = self.get_tissue_parts_indices(
#                     x_tissue, patch_size_scaled * self.min_consec_axis_info
#                 )

#                 # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
#                 for xidx in x_tissue_parts_indices:
#                     x_tissue_subparts_coords = self.get_tissue_subparts_coords(
#                         xidx, patch_size_scaled, self.min_decimal_keep
#                     )

#                     for x in x_tissue_subparts_coords:
#                         coords = self.eval_and_append_xy_coords(
#                             coords,
#                             image,
#                             mask,
#                             patch_size_scaled,
#                             x,
#                             y,
#                         )

#         if len(coords) < 1:
#             print("Not enough tissue in image (x-dim)", RuntimeWarning)
#             return [(0, 0, 0)]

#         np.random.shuffle(coords)
#         for y, x in coords[: self.max_n_samples]:
#             yield int(x / downscale_factor), int(y / downscale_factor)

#     def mask_tissue(self, image, kernel_size=(7, 7), gray_threshold=220):
#         """Masks tissue in image. Uses gray-scaled image, as well as
#         dilation kernels and 'gap filling'
#         """
#         # Define elliptic kernel
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
#         # Convert rgb to gray scale for easier masking
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         # Now mask the gray-scaled image (capturing tissue in biopsy)
#         mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
#         # Use dilation and findContours to fill in gaps/holes in masked tissue
#         mask = cv2.dilate(mask, kernel, iterations=1)
#         contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contour:
#             cv2.drawContours(mask, [cnt], 0, 1, -1)
#         return mask

#     def transpose_image(self, image: np.ndarray):
#         """Inputs an image and transposes it, accepts
#         both 2-d (mask) and 3-d (rgb image) arrays
#         """
#         if image is None:
#             return None
#         elif image.ndim == 2:
#             return np.transpose(image, (1, 0)).copy()
#         elif image.ndim == 3:
#             return np.transpose(image, (1, 0, 2)).copy()
#         return None

#     def get_tissue_parts_indices(self, tissue, min_consec_info):
#         """If there are multiple tissue parts in 'tissue', 'tissue' will be
#         split. Each tissue part will be taken care of separately (later on),
#         and if the tissue part is less than min_consec_info, it's considered
#         to small and won't be returned.
#         """
#         split_points = np.where(np.diff(tissue) != 1)[0] + 1
#         tissue_parts = np.split(tissue, split_points)
#         return [tp for tp in tissue_parts if len(tp) >= min_consec_info]

#     def get_tissue_subparts_coords(self, subtissue, patch_size, min_decimal_keep):
#         """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
#         This tissue part is divided into N subparts and returned.
#         Argument min_decimal_keep basically decides if we should reduce the
#         N subparts to N-1 subparts, due to overflow.
#         """
#         start, end = subtissue[0], subtissue[-1]
#         num_subparts = (end - start) / patch_size
#         if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
#             num_subparts = math.floor(num_subparts)
#         else:
#             num_subparts = math.ceil(num_subparts)

#         excess = (num_subparts * patch_size) - (end - start)
#         shift = excess // 2

#         return [i * patch_size + start - shift for i in range(num_subparts)]

#     def eval_and_append_xy_coords(
#         self,
#         coords,
#         image,
#         mask,
#         patch_size,
#         x,
#         y,
#     ):
#         """Based on computed x and y coordinates of patch:
#         slices out patch from original image, flattens it,
#         preprocesses it, and finally evaluates its mask.
#         If patch contains more info than min_patch_info,
#         the patch coordinates are kept, along with a value
#         'val1' that estimates how much information there
#         is in the patch. Smaller 'val1' assumes more info.
#         """
#         patch_1d = image[y : y + patch_size, x : x + patch_size, :].mean(axis=2).reshape(-1)
#         idx_tissue = np.where(patch_1d <= 210)[0]

#         if len(idx_tissue) > 0:
#             val = mask[y : y + patch_size, x : x + patch_size].mean()
#             if val > self.min_patch_info:
#                 coords = np.concatenate([coords, [[y, x]]])

#         return coords
