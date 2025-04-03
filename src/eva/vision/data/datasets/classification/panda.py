"""PANDA dataset class."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import pandas as pd
import torch
from torchvision import tv_tensors
from torchvision.datasets import utils
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.core.data import splitting
from eva.vision.data.datasets import _validators, structs, vision, wsi
from eva.vision.data.wsi.patching import samplers


class PANDA(wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for PANDA images and corresponding targets."""

    _train_split_ratio: float = 0.7
    """Train split ratio."""

    _val_split_ratio: float = 0.15
    """Validation split ratio."""

    _test_split_ratio: float = 0.15
    """Test split ratio."""

    _resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="train_with_noisy_labels.csv",
            url="https://raw.githubusercontent.com/analokmaus/kaggle-panda-challenge-public/master/train.csv",
            md5="5e4bfc78bda9603d2e2faf3ed4b21dfa",
        )
    ]
    """Download resources."""

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        split: Literal["train", "val", "test"] | None = None,
        width: int = 224,
        height: int = 224,
        target_mpp: float = 0.5,
        backend: str = "openslide",
        image_transforms: Callable | None = None,
        coords_path: str | None = None,
        seed: int = 42,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
            split: Dataset split to use. If `None`, the entire dataset is used.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            backend: The backend to use for reading the whole-slide images.
            image_transforms: Transforms to apply to the extracted image patches.
            coords_path: File path to save the patch coordinates as .csv.
            seed: Random seed for reproducibility.
        """
        self._split = split
        self._root = root
        self._seed = seed

        self._download_resources()

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=target_mpp,
            backend=backend,
            image_transforms=image_transforms,
            coords_path=coords_path,
        )

    @property
    @override
    def classes(self) -> List[str]:
        return ["0", "1", "2", "3", "4", "5"]

    @functools.cached_property
    def annotations(self) -> pd.DataFrame:
        """Loads the dataset labels."""
        path = os.path.join(self._root, "train_with_noisy_labels.csv")
        return pd.read_csv(path, index_col="image_id")

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, False)

        if not os.path.isdir(os.path.join(self._root, "train_images")):
            raise FileNotFoundError("'train_images' directory not found in the root folder.")
        if not os.path.isfile(os.path.join(self._root, "train_with_noisy_labels.csv")):
            raise FileNotFoundError("'train.csv' file not found in the root folder.")

    def _download_resources(self) -> None:
        """Downloads the dataset resources."""
        for resource in self._resources:
            utils.download_url(resource.url, self._root, resource.filename, resource.md5)

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=None,
            n_classes=6,
            first_and_last_labels=("0", "5"),
        )

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, torch.Tensor, Dict[str, Any]]:
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        file_path = self._file_paths[self._get_dataset_idx(index)]
        return torch.tensor(self._get_target_from_path(file_path), dtype=torch.int64)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return wsi.MultiWsiDataset.load_metadata(self, index)

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        image_dir = os.path.join(self._root, "train_images")
        file_paths = sorted(glob.glob(os.path.join(image_dir, "*.tiff")))
        file_paths = [os.path.relpath(path, self._root) for path in file_paths]
        if len(file_paths) != len(self.annotations):
            raise ValueError(
                f"Expected {len(self.annotations)} images, found {len(file_paths)} in {image_dir}."
            )
        file_paths = self._filter_noisy_labels(file_paths)
        targets = [self._get_target_from_path(file_path) for file_path in file_paths]

        train_indices, val_indices, test_indices = splitting.stratified_split(
            samples=file_paths,
            targets=targets,
            train_ratio=self._train_split_ratio,
            val_ratio=self._val_split_ratio,
            test_ratio=self._test_split_ratio,
            seed=self._seed,
        )

        match split:
            case "train":
                return [file_paths[i] for i in train_indices]
            case "val":
                return [file_paths[i] for i in val_indices]
            case "test":
                return [file_paths[i] for i in test_indices or []]
            case None:
                return file_paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val', 'test' or `None`.")

    def _filter_noisy_labels(self, file_paths: List[str]):
        is_noisy_filter = self.annotations["noise_ratio_10"] == 0
        non_noisy_image_ids = set(self.annotations.loc[~is_noisy_filter].index)
        filtered_file_paths = [
            file_path
            for file_path in file_paths
            if self._get_id_from_path(file_path) in non_noisy_image_ids
        ]
        return filtered_file_paths

    def _get_target_from_path(self, file_path: str) -> int:
        return self.annotations.loc[self._get_id_from_path(file_path), "isup_grade"]

    def _get_id_from_path(self, file_path: str) -> str:
        return os.path.basename(file_path).replace(".tiff", "")


class PANDASmall(PANDA):
    """Small version of the PANDA dataset for quicker benchmarking."""

    _train_split_ratio: float = 0.1
    """Train split ratio."""

    _val_split_ratio: float = 0.05
    """Validation split ratio."""

    _test_split_ratio: float = 0.05
    """Test split ratio."""
