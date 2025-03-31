"""Dataset classes for whole-slide images."""

import bisect
import os
from typing import Any, Callable, Dict, List

import pandas as pd
from loguru import logger
from torch.utils.data import dataset as torch_datasets
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.core.data.datasets import base
from eva.vision.data import wsi
from eva.vision.data.wsi.patching import samplers


class WsiDataset(base.MapDataset):
    """Dataset class for reading patches from whole-slide images."""

    def __init__(
        self,
        file_path: str,
        width: int,
        height: int,
        sampler: samplers.Sampler,
        target_mpp: float,
        overwrite_mpp: float | None = None,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
    ):
        """Initializes a new dataset instance.

        Args:
            file_path: Path to the whole-slide image file.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            sampler: The sampler to use for sampling patch coordinates.
            target_mpp: Target microns per pixel (mpp) for the patches.
            overwrite_mpp: The microns per pixel (mpp) value to use when missing in WSI metadata.
            backend: The backend to use for reading the whole-slide images.
            image_transforms: Transforms to apply to the extracted image patches.
        """
        super().__init__()

        self._file_path = file_path
        self._width = width
        self._height = height
        self._sampler = sampler
        self._target_mpp = target_mpp
        self._overwrite_mpp = overwrite_mpp
        self._backend = backend
        self._image_transforms = image_transforms

    @override
    def __len__(self):
        return len(self._coords.x_y)

    def filename(self, index: int) -> str:
        """Returns the filename of the patch at the specified index."""
        return f"{self._file_path}_{index}"

    @property
    def _wsi(self) -> wsi.Wsi:
        return wsi.get_cached_wsi(self._file_path, self._backend, self._overwrite_mpp)

    @property
    def _coords(self) -> wsi.PatchCoordinates:
        return wsi.get_cached_coords(
            file_path=self._file_path,
            width=self._width,
            height=self._height,
            target_mpp=self._target_mpp,
            overwrite_mpp=self._overwrite_mpp,
            sampler=self._sampler,
            backend=self._backend,
        )

    @override
    def __getitem__(self, index: int) -> tv_tensors.Image:
        x, y = self._coords.x_y[index]
        width, height, level_idx = self._coords.width, self._coords.height, self._coords.level_idx
        patch = self._wsi.read_region((x, y), level_idx, (width, height))
        patch = functional.to_image(patch)
        patch = self._apply_transforms(patch)
        return patch

    def load_metadata(self, index: int) -> Dict[str, Any]:
        """Loads the metadata for the patch at the specified index."""
        x, y = self._coords.x_y[index]
        return {
            "x": x,
            "y": y,
            "width": self._coords.width,
            "height": self._coords.height,
            "level_idx": self._coords.level_idx,
        }

    def _apply_transforms(self, image: tv_tensors.Image) -> tv_tensors.Image:
        if self._image_transforms is not None:
            image = self._image_transforms(image)
        return image


class MultiWsiDataset(base.MapDataset):
    """Dataset class for reading patches from multiple whole-slide images."""

    def __init__(
        self,
        root: str,
        file_paths: List[str],
        width: int,
        height: int,
        sampler: samplers.Sampler,
        target_mpp: float,
        overwrite_mpp: float | None = None,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
        coords_path: str | None = None,
    ):
        """Initializes a new dataset instance.

        Args:
            root: Root directory of the dataset.
            file_paths: List of paths to the whole-slide image files, relative to the root.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            overwrite_mpp: The microns per pixel (mpp) value to use when missing in WSI metadata.
            sampler: The sampler to use for sampling patch coordinates.
            backend: The backend to use for reading the whole-slide images.
            image_transforms: Transforms to apply to the extracted image patches.
            coords_path: File path to save the patch coordinates as .csv.
        """
        super().__init__()

        self._root = root
        self._file_paths = file_paths
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._overwrite_mpp = overwrite_mpp
        self._sampler = sampler
        self._backend = backend
        self._image_transforms = image_transforms
        self._coords_path = coords_path

        self._concat_dataset: torch_datasets.ConcatDataset

    @property
    def datasets(self) -> List[WsiDataset]:
        """Returns the list of WSI datasets."""
        return self._concat_dataset.datasets  # type: ignore

    @property
    def cumulative_sizes(self) -> List[int]:
        """Returns the cumulative sizes of the WSI datasets."""
        return self._concat_dataset.cumulative_sizes

    @override
    def configure(self) -> None:
        self._concat_dataset = torch_datasets.ConcatDataset(datasets=self._load_datasets())
        self._save_coords_to_file()

    @override
    def __len__(self) -> int:
        return len(self._concat_dataset)

    @override
    def __getitem__(self, index: int) -> tv_tensors.Image:
        return self._concat_dataset[index]

    def filename(self, index: int) -> str:
        """Returns the filename of the patch at the specified index."""
        return os.path.basename(self._file_paths[self._get_dataset_idx(index)])

    def load_metadata(self, index: int) -> Dict[str, Any]:
        """Loads the metadata for the patch at the specified index."""
        dataset_index, sample_index = self._get_dataset_idx(index), self._get_sample_idx(index)
        patch_metadata = self.datasets[dataset_index].load_metadata(sample_index)
        return {"wsi_id": self.filename(index).split(".")[0]} | patch_metadata

    def _load_datasets(self) -> list[WsiDataset]:
        logger.info(f"Initializing dataset with {len(self._file_paths)} WSIs ...")
        wsi_datasets = []
        for file_path in self._file_paths:
            file_path = (
                os.path.join(self._root, file_path) if self._root not in file_path else file_path
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            wsi_datasets.append(
                WsiDataset(
                    file_path=file_path,
                    width=self._width,
                    height=self._height,
                    sampler=self._sampler,
                    target_mpp=self._target_mpp,
                    overwrite_mpp=self._overwrite_mpp,
                    backend=self._backend,
                    image_transforms=self._image_transforms,
                )
            )
        return wsi_datasets

    def _get_dataset_idx(self, index: int) -> int:
        return bisect.bisect_right(self.cumulative_sizes, index)

    def _get_sample_idx(self, index: int) -> int:
        dataset_idx = self._get_dataset_idx(index)
        return index if dataset_idx == 0 else index - self.cumulative_sizes[dataset_idx - 1]

    def _save_coords_to_file(self):
        if self._coords_path is not None:
            coords = [
                {"file": self._file_paths[i]} | dataset._coords.to_dict()
                for i, dataset in enumerate(self.datasets)
            ]
            os.makedirs(os.path.abspath(os.path.join(self._coords_path, os.pardir)), exist_ok=True)
            pd.DataFrame(coords).to_csv(self._coords_path, index=False)
            logger.info(f"Saved patch coordinates to: {self._coords_path}")
