"""Dataset classes for whole-slide images."""

import bisect
import os
from typing import Callable, List

import numpy as np
from loguru import logger
from torch.utils.data import dataset as torch_datasets
from typing_extensions import override

from eva.core.data.datasets import base
from eva.vision.data import wsi
from eva.vision.data.datasets import vision
from eva.vision.data.wsi.patching import samplers


class WsiDataset(vision.VisionDataset):
    """Dataset class for reading patches from whole-slide images."""

    def __init__(
        self,
        file_path: str,
        width: int,
        height: int,
        target_mpp: float,
        sampler: samplers.Sampler,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
    ):
        """Initializes a new dataset instance.

        Args:
            file_path: Path to the whole-slide image file.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
            image_transforms: Transforms to apply to the extracted image patches.
        """
        self._file_path = file_path
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._sampler = sampler
        self._backend = backend
        self._image_transforms = image_transforms

    @override
    def __len__(self):
        return len(self._coords.x_y)

    @override
    def filename(self, index: int) -> str:
        return f"{self._file_path}_{index}"

    @property
    def _wsi(self) -> wsi.Wsi:
        return wsi.get_cached_wsi(self._file_path, self._backend)

    @property
    def _coords(self) -> wsi.PatchCoordinates:
        return wsi.get_cached_coords(
            file_path=self._file_path,
            width=self._width,
            height=self._height,
            target_mpp=self._target_mpp,
            sampler=self._sampler,
            backend=self._backend,
        )

    @override
    def __getitem__(self, index: int) -> np.ndarray:
        x, y = self._coords.x_y[index]
        width, height, level_idx = self._coords.width, self._coords.height, self._coords.level_idx
        patch = self._wsi.read_region((x, y), level_idx, (width, height))
        patch = self._apply_transforms(patch)
        return patch

    def _apply_transforms(self, image: np.ndarray) -> np.ndarray:
        if self._image_transforms is not None:
            image = self._image_transforms(image)
        return image


class MultiWsiDataset(torch_datasets.ConcatDataset, base.Dataset):
    """Dataset class for reading patches from multiple whole-slide images."""

    def __init__(
        self,
        root: str,
        file_paths: List[str],
        width: int,
        height: int,
        target_mpp: float,
        sampler: samplers.Sampler,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
    ):
        """Initializes a new dataset instance.

        Args:
            root: Root directory of the dataset.
            file_paths: List of paths to the whole-slide image files, relative to the root.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
            image_transforms: Transforms to apply to the extracted image patches.
        """
        self._root = root
        self._file_paths = file_paths
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._sampler = sampler
        self._backend = backend
        self._image_transforms = image_transforms

    @override
    def configure(self) -> None:
        super().__init__(self._load_datasets())

    def _load_datasets(self) -> list[WsiDataset]:
        logger.info(f"Initializing dataset with {len(self._file_paths)} WSIs ...")
        wsi_datasets = []
        for file_path in self._file_paths:
            file_path = os.path.join(self._root, file_path) if self._root else file_path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            wsi_datasets.append(
                WsiDataset(
                    file_path=file_path,
                    width=self._width,
                    height=self._height,
                    target_mpp=self._target_mpp,
                    sampler=self._sampler,
                    backend=self._backend,
                    image_transforms=self._image_transforms,
                )
            )
        return wsi_datasets

    def _get_dataset_idx(self, index: int) -> int:
        return bisect.bisect_right(self.cumulative_sizes, index)
