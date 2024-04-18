"""Dataset classes for whole-slide images."""

import os
from functools import cached_property
from typing import Callable, Dict

import pandas as pd
import torch
from torch.utils.data import dataset as torch_datasets
from typing_extensions import override

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
        backend: wsi.WsiBackend = wsi.WsiBackend.OPENSLIDE,
        transforms: Callable[..., torch.Tensor] | None = None,
    ):
        """Initializes a new dataset instance.

        Args:
            file_path: Path to the whole-slide image file.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
            transforms: Transforms to apply to the extracted patch tensors.
        """
        self._file_path = file_path
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._backend = backend
        self._sampler = sampler
        self._transforms = transforms

    @override
    def __len__(self):
        return len(self._coords.x_y)

    @override
    def filename(self, index: int) -> str:
        return f"{self._file_path}_{index}"

    @cached_property
    def _wsi(self) -> wsi.Wsi:
        wsi_obj = wsi.get_wsi_class(self._backend)(self._file_path)
        wsi_obj.open_slide()
        return wsi_obj

    @cached_property
    def _coords(self) -> wsi.PatchCoordinates:
        return wsi.PatchCoordinates.from_file(
            wsi_path=self._file_path,
            width=self._width,
            height=self._height,
            target_mpp=self._target_mpp,
            backend=self._backend,
            sampler=self._sampler,
        )

    @override
    def __getitem__(self, index: int) -> torch.Tensor:
        x, y = self._coords.x_y[index]
        width, height, level_idx = self._coords.width, self._coords.height, self._coords.level_idx
        patch = self._wsi.read_region((x, y), (width, height), level_idx)
        patch = self._apply_transforms(torch.from_numpy(patch))
        return patch

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._transforms:
            tensor = self._transforms(tensor)
        return tensor


class MultiWsiDataset(torch_datasets.ConcatDataset, vision.VisionDataset):
    """Dataset class for reading patches from multiple whole-slide images."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
    }

    def __init__(
        self,
        root: str,
        manifest_file: str,
        width: int,
        height: int,
        target_mpp: float,
        sampler: samplers.Sampler,
        backend: wsi.WsiBackend = wsi.WsiBackend.OPENSLIDE,
        transforms: Callable | None = None,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        """Initializes a new dataset instance.

        Args:
            root: Root directory of the dataset.
            manifest_file: The path to the manifest file, which is relative to
                the `root` argument.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
            transforms: Transforms to apply to the extracted patch tensors.
            column_mapping: Defines the map between the variables and the manifest
                columns. It will overwrite the `default_column_mapping` with
                the provided values, so that `column_mapping` can contain only the
                values which are altered or missing.
        """
        self._root = root
        self._manifest_file = manifest_file
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._sampler = sampler
        self._backend = backend
        self._transforms = transforms
        self._column_mapping = column_mapping

        self._manifest = self._load_manifest(os.path.join(self._root, self._manifest_file))
        super().__init__(self._load_datasets())

    def _load_datasets(self) -> list[WsiDataset]:
        wsi_datasets = []
        for _, row in self._manifest.iterrows():
            file_path = os.path.join(self._root, str(row[self._column_mapping["path"]]))
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
                    transforms=self._transforms,
                )
            )
        return wsi_datasets

    def _load_manifest(self, manifest_path: str) -> pd.DataFrame:
        df = pd.read_csv(manifest_path)

        missing_columns = set(self._column_mapping.values()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in the manifest file: {missing_columns}")

        return df
