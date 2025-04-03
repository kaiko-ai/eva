"""Camelyon16 dataset class."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import pandas as pd
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision, wsi
from eva.vision.data.wsi.patching import samplers


class Camelyon16(wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for Camelyon16 images and corresponding targets."""

    _val_slides = [
        "normal_010",
        "normal_013",
        "normal_016",
        "normal_017",
        "normal_019",
        "normal_020",
        "normal_025",
        "normal_030",
        "normal_031",
        "normal_032",
        "normal_052",
        "normal_056",
        "normal_057",
        "normal_067",
        "normal_076",
        "normal_079",
        "normal_085",
        "normal_095",
        "normal_098",
        "normal_099",
        "normal_101",
        "normal_102",
        "normal_105",
        "normal_106",
        "normal_109",
        "normal_129",
        "normal_132",
        "normal_137",
        "normal_142",
        "normal_143",
        "normal_148",
        "normal_152",
        "tumor_001",
        "tumor_005",
        "tumor_011",
        "tumor_012",
        "tumor_013",
        "tumor_019",
        "tumor_031",
        "tumor_037",
        "tumor_043",
        "tumor_046",
        "tumor_057",
        "tumor_065",
        "tumor_069",
        "tumor_071",
        "tumor_073",
        "tumor_079",
        "tumor_080",
        "tumor_081",
        "tumor_082",
        "tumor_085",
        "tumor_097",
        "tumor_109",
    ]
    """Validation slide names, same as the ones in patch camelyon."""

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
        self._width = width
        self._height = height
        self._target_mpp = target_mpp
        self._seed = seed

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
        return ["normal", "tumor"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"normal": 0, "tumor": 1}

    @functools.cached_property
    def annotations_test_set(self) -> Dict[str, str]:
        """Loads the dataset labels."""
        path = os.path.join(self._root, "testing/reference.csv")
        reference_df = pd.read_csv(path, header=None)
        return {k: v.lower() for k, v in reference_df[[0, 1]].itertuples(index=False)}

    @functools.cached_property
    def annotations(self) -> Dict[str, str]:
        """Loads the dataset labels."""
        annotations = {}
        if self._split in ["test", None]:
            path = os.path.join(self._root, "testing/reference.csv")
            reference_df = pd.read_csv(path, header=None)
            annotations.update(
                {k: v.lower() for k, v in reference_df[[0, 1]].itertuples(index=False)}
            )

        if self._split in ["train", "val", None]:
            annotations.update(
                {
                    self._get_id_from_path(file_path): self._get_class_from_path(file_path)
                    for file_path in self._file_paths
                    if "test" not in file_path
                }
            )
        return annotations

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, False)

        expected_directories = ["training/normal", "training/tumor", "testing/images"]
        for resource in expected_directories:
            if not os.path.isdir(os.path.join(self._root, resource)):
                raise FileNotFoundError(f"'{resource}' not found in the root folder.")

        if not os.path.isfile(os.path.join(self._root, "testing/reference.csv")):
            raise FileNotFoundError("'reference.csv' file not found in the testing folder.")

    @override
    def validate(self) -> None:

        expected_n_files = {
            "train": 216,
            "val": 54,
            "test": 129,
            None: 399,
        }
        _validators.check_number_of_files(
            self._file_paths, expected_n_files[self._split], self._split
        )
        _validators.check_dataset_integrity(
            self,
            length=None,
            n_classes=2,
            first_and_last_labels=("normal", "tumor"),
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
        class_name = self.annotations[self._get_id_from_path(file_path)]
        return torch.tensor(self.class_to_idx[class_name], dtype=torch.int64)

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return wsi.MultiWsiDataset.load_metadata(self, index)

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        train_paths, val_paths = [], []
        for path in glob.glob(os.path.join(self._root, "training/**/*.tif")):
            if self._get_id_from_path(path) in self._val_slides:
                val_paths.append(path)
            else:
                train_paths.append(path)
        test_paths = glob.glob(os.path.join(self._root, "testing/images", "*.tif"))

        match split:
            case "train":
                paths = train_paths
            case "val":
                paths = val_paths
            case "test":
                paths = test_paths
            case None:
                paths = train_paths + val_paths + test_paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val' or `None`.")
        return sorted([os.path.relpath(path, self._root) for path in paths])

    def _get_id_from_path(self, file_path: str) -> str:
        """Extracts the slide ID from the file path."""
        return os.path.basename(file_path).replace(".tif", "")

    def _get_class_from_path(self, file_path: str) -> str:
        """Extracts the class name from the file path."""
        class_name = self._get_id_from_path(file_path).split("_")[0]
        if class_name not in self.classes:
            raise ValueError(f"Invalid class name '{class_name}' in file path '{file_path}'.")
        return class_name
