"""BCSS dataset."""

import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision, wsi
from eva.vision.data.datasets.segmentation import _utils
from eva.vision.data.wsi.patching import samplers
from eva.vision.utils import io


class BCSS(wsi.MultiWsiDataset, vision.VisionDataset[tv_tensors.Image, tv_tensors.Mask]):
    """Dataset class for BCSS semantic segmentation task.

    Source: https://github.com/PathologyDataScience/BCSS

    We apply the the class grouping proposed by the challenge baseline:
    https://bcsegmentation.grand-challenge.org/Baseline/

    outside_roi: outside_roi
    tumor: angioinvasion, dcis
    stroma: stroma
    inflammatory: lymphocytic_infiltrate, plasma_cells, other_immune_infiltrate
    necrosis: necrosis_or_debris
    other: remaining

    Be aware that outside_roi should be assigned zero-weight during model training.
    """

    _train_split_ratio: float = 0.8
    """Train split ratio."""

    _val_split_ratio: float = 0.2
    """Validation split ratio."""

    _expected_length: int = 151
    """Expected dataset length."""

    _val_institutes = {"BH", "C8", "A8", "A1", "E9"}
    """Medical institutes to use for the validation split."""

    _test_institutes = {"OL", "LL", "E2", "EW", "GM", "S3"}
    """Medical institutes to use for the test split."""

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        split: Literal["train", "val", "trainval", "test"] | None = None,
        width: int = 224,
        height: int = 224,
        target_mpp: float = 0.5,
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory of the dataset.
            sampler: The sampler to use for sampling patch coordinates.
                If `None`, it will use the ::class::`GridSampler` sampler.
            split: Dataset split to use. If `None`, the entire dataset is used.
            width: Width of the patches to be extracted, in pixels.
            height: Height of the patches to be extracted, in pixels.
            target_mpp: Target microns per pixel (mpp) for the patches.
            transforms: Transforms to apply to the extracted image & mask patches.
        """
        self._split = split
        self._root = root

        self.datasets: List[wsi.WsiDataset]  # type: ignore

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler or samplers.GridSampler(max_samples=1000),
            target_mpp=target_mpp,
            overwrite_mpp=0.25,
            backend="pil",
        )
        vision.VisionDataset.__init__(self, transforms=transforms)

    @property
    @override
    def classes(self) -> List[str]:
        return list(self.class_to_idx.keys())

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {
            "outside_roi": 0,
            "tumor": 1,
            "stroma": 2,
            "inflammatory": 3,
            "necrosis": 4,
            "other": 5,
        }

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

        if not os.path.isdir(os.path.join(self._root, "masks")):
            raise FileNotFoundError(f"'masks' directory not found in {self._root}.")
        if not os.path.isdir(os.path.join(self._root, "rgbs_colorNormalized")):
            raise FileNotFoundError(f"'rgbs_colorNormalized' directory not found in {self._root}.")

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=None,
            n_classes=6,
            first_and_last_labels=((self.classes[0], self.classes[-1])),
        )

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask, Dict[str, Any]]:
        return vision.VisionDataset.__getitem__(self, index)

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        path = self._get_mask_path(index)
        mask = io.read_image_as_array(path)
        mask_patch = _utils.extract_mask_patch(mask, self, index)
        mask_patch = self._map_classes(mask_patch)
        return tv_tensors.Mask(mask_patch, dtype=torch.int64)  # type: ignore[reportCallIssue]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        (x, y), width, height = _utils.get_coords_at_index(self, index)
        return {"coords": f"{x},{y},{width},{height}"}

    def _load_file_paths(
        self, split: Literal["train", "val", "trainval", "test"] | None = None
    ) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        file_paths = sorted(glob.glob(os.path.join(self._root, "rgbs_colorNormalized/*.png")))
        if len(file_paths) != self._expected_length:
            raise ValueError(
                f"Expected {self._expected_length} images, found {len(file_paths)} in {self._root}."
            )

        train_indices, val_indices, test_indices = [], [], []
        for i, path in enumerate(file_paths):
            institute = Path(path).stem.split("-")[1]
            if institute in self._test_institutes:
                test_indices.append(i)
            elif institute in self._val_institutes:
                val_indices.append(i)
            else:
                train_indices.append(i)

        match split:
            case "train":
                return [file_paths[i] for i in train_indices]
            case "val":
                return [file_paths[i] for i in val_indices]
            case "trainval":
                return [file_paths[i] for i in train_indices + val_indices]
            case "test":
                return [file_paths[i] for i in test_indices]
            case None:
                return file_paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val', 'test' or `None`.")

    def _get_mask_path(self, index):
        """Returns the path to the mask file corresponding to the patch at the given index."""
        return os.path.join(self._root, "masks", self.filename(index))

    def _map_classes(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Maps the classes of the mask array to the grouped tissue type classes."""
        original_to_grouped_class_mapping = {
            "outside_roi": "outside_roi",
            "angioinvasion": "tumor",
            "dcis": "tumor",
            "stroma": "stroma",
            "lymphocytic_infiltrate": "inflammatory",
            "plasma_cells": "inflammatory",
            "other_immune_infiltrate": "inflammatory",
            "necrosis_or_debris": "necrosis",
        }

        mapped_array = np.full_like(array, fill_value=self.class_to_idx["other"], dtype=int)

        for original_class, grouped_class in original_to_grouped_class_mapping.items():
            original_class_idx = _original_class_to_idx[original_class]
            grouped_class_idx = self.class_to_idx[grouped_class]
            mapped_array[array == original_class_idx] = grouped_class_idx

        return mapped_array


_original_class_to_idx = {
    "outside_roi": 0,
    "tumor": 1,
    "stroma": 2,
    "lymphocytic_infiltrate": 3,
    "necrosis_or_debris": 4,
    "glandular_secretions": 5,
    "blood": 6,
    "exclude": 7,
    "metaplasia_NOS": 8,
    "fat": 9,
    "plasma_cells": 10,
    "other_immune_infiltrate": 11,
    "mucoid_material": 12,
    "normal_acinus_or_duct": 13,
    "lymphatics": 14,
    "undetermined": 15,
    "nerve": 16,
    "skin_adnexa": 17,
    "blood_vessel": 18,
    "angioinvasion": 19,
    "dcis": 20,
    "other": 21,
}
