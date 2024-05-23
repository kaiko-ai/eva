"""Camelyon16 dataset class."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from typing_extensions import override

from eva.vision.data.datasets import _validators, wsi
from eva.vision.data.datasets.classification import base
from eva.vision.data.wsi.patching import samplers


class Camelyon16(wsi.MultiWsiDataset, base.ImageClassification):
    """Dataset class for Camelyon16 images and corresponding targets."""

    _val_slides = [
        'normal_010',
        'normal_013',
        'normal_016',
        'normal_017',
        'normal_019',
        'normal_020',
        'normal_025',
        'normal_030',
        'normal_031',
        'normal_032',
        'normal_052',
        'normal_056',
        'normal_057',
        'normal_067',
        'normal_076',
        'normal_079',
        'normal_085',
        'normal_095',
        'normal_098',
        'normal_099',
        'normal_101',
        'normal_102',
        'normal_105',
        'normal_106',
        'normal_109',
        'normal_129',
        'normal_132',
        'normal_137',
        'normal_142',
        'normal_143',
        'normal_148',
        'normal_152',
        'tumor_001',
        'tumor_005',
        'tumor_011',
        'tumor_012',
        'tumor_013',
        'tumor_019',
        'tumor_031',
        'tumor_037',
        'tumor_043',
        'tumor_046',
        'tumor_057',
        'tumor_065',
        'tumor_069',
        'tumor_071',
        'tumor_073',
        'tumor_079',
        'tumor_080',
        'tumor_081',
        'tumor_082',
        'tumor_085',
        'tumor_097',
        'tumor_109',
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
        )

    @property
    @override
    def classes(self) -> List[str]:
        return ["0", "1"]

    @functools.cached_property
    def test_annotations(self) -> pd.DataFrame:
        """Loads the dataset labels."""
        path = os.path.join(self._root, "testing/reference.csv")
        reference_df = pd.read_csv(path, header=None)
        return {k: v.lower() for k, v in reference_df[[0, 1]].itertuples(index=False)}

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

        if not os.path.isdir(os.path.join(self._root, "training/normal")):
            raise FileNotFoundError("'training/normal' directory not found in the root folder.")
        if not os.path.isdir(os.path.join(self._root, "training/tumor")):
            raise FileNotFoundError("'training/tumor' directory not found in the root folder.")
        if not os.path.isdir(os.path.join(self._root, "testing/images")):
            raise FileNotFoundError("'testing/images' directory not found in the root folder.")
        if not os.path.isfile(os.path.join(self._root, "testing/reference.csv")):
            raise FileNotFoundError("'reference.csv' file not found in the testing folder.")

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=None,
            n_classes=2,
            first_and_last_labels=("0", "1"),
        )

    @override
    def filename(self, index: int) -> str:
        return os.path.basename(self._file_paths[self._get_dataset_idx(index)])

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return base.ImageClassification.__getitem__(self, index)

    @override
    def load_image(self, index: int) -> torch.Tensor:
        return wsi.MultiWsiDataset.__getitem__(self, index)

    @override
    def load_target(self, index: int) -> np.ndarray:
        file_path = self._file_paths[self._get_dataset_idx(index)]
        return np.asarray(self._get_target_from_path(file_path))

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return {"wsi_id": self.filename(index).split(".")[0]}

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        train_images_paths_normal = sorted(
            glob.glob(os.path.join(self._root, "training/normal/*.tif"))
        )
        train_images_paths_tumor = sorted(
            glob.glob(os.path.join(self._root, "training/tumor/*.tif"))
        )
        train_images_paths_all = train_images_paths_normal + train_images_paths_tumor
        
        val_images_paths = [path for path in train_images_paths_all if self._get_id_from_path(path) in self._val_slides]
        train_images_paths = [path for path in train_images_paths_all if path not in val_images_paths]
        test_file_paths = sorted(glob.glob(os.path.join(self._root, "testing/images", "*.tif")))

        match split:
            case "train":
                paths = train_images_paths
            case "val":
                paths = val_images_paths
            case "test":
                paths = test_file_paths
            case None:
                paths = train_images_paths + test_file_paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val' or `None`.")
        return [os.path.relpath(path, self._root) for path in paths]

    def _get_target_from_path(self, file_path: str) -> int:
        """Returns the target label based on the file path."""
        slide_id = self._get_id_from_path(file_path)
        test_annotation = self.test_annotations.get(slide_id, "")
        if "normal" in file_path or "normal" in test_annotation:
            return 0
        elif "tumor" in file_path or "tumor" in test_annotation:
            return 1
        else:
            raise ValueError(f"Invalid file path: {file_path}")

    def _get_id_from_path(self, file_path: str) -> str:
        """Extracts the slide ID from the file path."""
        return os.path.basename(file_path).replace(".tif", "")
