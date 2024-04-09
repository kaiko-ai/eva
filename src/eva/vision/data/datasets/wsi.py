import os
import random
from functools import cached_property
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import dataset as torch_datasets
from typing_extensions import override

from eva.vision.data import wsi
from eva.vision.data.datasets import vision


class WsiDataset(vision.VisionDataset):
    def __init__(
        self,
        file_path: str,
        n_samples: int,
        width: int,
        height: int,
        target_mpp: float,
        backend: wsi.WsiBackend = wsi.WsiBackend.OPENSLIDE,
        transforms: Callable | None = None,
        dataset_idx: int = 0,
    ):
        """Args:
        file_path: Path to the whole-slide image file.
        n_samples: Number of patches to sample from each slide.
        width: Width of the patches to be extracted, in pixels.
        height: Height of the patches to be extracted, in pixels.
        target_mpp: Target microns per pixel (mpp) for the patches.
        backend: The backend to use for reading the whole-slide images.
        transforms: A function that takes in an image and returns a transformed version.
        dataset_idx: Index of the dataset, useful when using in combination with ConcatDataset.
        """
        self.dataset_idx = dataset_idx

        self._file_path = file_path
        self._n_samples = n_samples
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._backend = backend
        self._transforms = transforms

    def __len__(self):
        return self._n_samples

    @override
    @property
    def filename(self, index: int) -> str:
        return f"{self._file_path}_{index}"

    @cached_property
    def _wsi(self) -> wsi.Wsi:
        wsi_obj = wsi.get_wsi_class(self._backend)(self._file_path)
        wsi_obj.open_slide()
        return wsi_obj

    def __getitem__(self, index: int) -> torch.Tensor:
        # Calculate the desired zoom level based on target_mpp
        level_idx, width, height = self._get_closest_level(self._wsi, self._target_mpp)

        # Random Sampling
        # TODO: make sampling method configurable
        # TODO: add support for masking of unwanted regions
        x_max, y_max = self._wsi.level_dimensions[level_idx]
        x = random.randint(0, x_max - width)
        y = random.randint(0, y_max - height)

        patch = self._wsi.read_region((x, y), (width, height), level_idx)
        patch = self._apply_transforms(patch)
        return patch

    def _get_closest_level(self, slide: wsi.Wsi, target_mpp: float):
        """Calculate the slide level closest to the target mpp."""
        # Calculate the mpp for each level
        level_mpps = slide.mpp * np.array(slide.level_downsamples)

        # Ignore levels with higher mpp
        level_mpps_filtered = level_mpps.copy()
        level_mpps_filtered[level_mpps_filtered > target_mpp] = 0

        if level_mpps_filtered.max() == 0:
            # When all levels have higher mpp than target_mpp return the level with lowest mpp
            level_idx = np.argmin(level_mpps)
        else:
            level_idx = np.argmax(level_mpps_filtered)

        # Calculate the width & height in pixels scaled to the selected level
        mpp_ratio = slide.mpp / level_mpps[level_idx]
        width, height = int(mpp_ratio * self._width), int(mpp_ratio * self._height)

        return level_idx, width, height

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._transforms:
            tensor = self._transforms(tensor)
        return tensor


class MultiWsiDataset(torch_datasets.ConcatDataset):
    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
    }

    def __init__(
        self,
        root: str,
        manifest_file: str,
        n_samples: int,
        width: int,
        height: int,
        target_mpp: float,
        backend: wsi.WsiBackend = wsi.WsiBackend.OPENSLIDE,
        transforms: Callable | None = None,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        self._root = root
        self._manifest_file = manifest_file
        self._n_samples = n_samples
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._backend = backend
        self._transforms = transforms
        self._column_mapping = column_mapping

        self._manifest = self._load_manifest(os.path.join(self._root, self._manifest_file))
        super().__init__(self._load_datasets())

    def _load_datasets(self) -> list[WsiDataset]:
        wsi_datasets = []
        for index, row in self._manifest.iterrows():
            file_path = os.path.join(self._root, row[self._column_mapping["path"]])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            wsi_datasets.append(
                WsiDataset(
                    file_path=file_path,
                    n_samples=self._n_samples,
                    width=self._width,
                    height=self._height,
                    target_mpp=self._target_mpp,
                    backend=self._backend,
                    transforms=self._transforms,
                    dataset_idx=index,
                )
            )
        return wsi_datasets

    def _load_manifest(self, manifest_path: str) -> pd.DataFrame:
        df = pd.read_csv(manifest_path)

        missing_columns = set(self._column_mapping.values()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in the manifest file: {missing_columns}")

        return df
