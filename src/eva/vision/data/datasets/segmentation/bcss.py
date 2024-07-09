"""BCSS dataset."""

import glob
import os
from typing import Callable, Dict, List, Literal, Tuple

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional
from typing_extensions import override

from eva.core.data import splitting
from eva.vision.data.datasets import _validators, wsi
from eva.vision.data.datasets.segmentation import _utils, base
from eva.vision.data.wsi.patching import samplers
from eva.vision.utils import io


class BCSS(wsi.MultiWsiDataset, base.ImageSegmentation):
    """Dataset class for BCSS semantic segmentation task.

    Source: https://github.com/PathologyDataScience/BCSS

    Todo:
    - Please be aware that zero pixels represent regions outside the region of interest
    (“don’t care” class) and should be assigned zero-weight during model training;
    they do NOT represent an “other” class.
    """

    _train_split_ratio: float = 0.6
    """Train split ratio."""

    _val_split_ratio: float = 0.2
    """Validation split ratio."""

    _test_split_ratio: float = 0.2
    """Test split ratio."""

    _expected_length: int = 67
    """Expected dataset length."""

    def __init__(
        self,
        root: str,
        sampler: samplers.Sampler,
        split: Literal["train", "val", "test"] | None = None,
        width: int = 224,
        height: int = 224,
        target_mpp: float = 0.5,
        transforms: Callable | None = None,
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
            transforms: Transforms to apply to the extracted image & mask patches.
            seed: Random seed for reproducibility.
        """
        self._split = split
        self._root = root
        self._seed = seed

        self.datasets: List[wsi.WsiDataset]  # type: ignore

        wsi.MultiWsiDataset.__init__(
            self,
            root=root,
            file_paths=self._load_file_paths(split),
            width=width,
            height=height,
            sampler=sampler,
            target_mpp=target_mpp,
            overwrite_mpp=0.25,
            backend="pil",
        )
        base.ImageSegmentation.__init__(self, transforms=transforms)

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
            n_classes=22,
            first_and_last_labels=((self.classes[0], self.classes[-1])),
        )

    @override
    def __getitem__(self, index: int) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
        return base.ImageSegmentation.__getitem__(self, index)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        image_array = wsi.MultiWsiDataset.__getitem__(self, index)
        return functional.to_image(image_array)

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        path = self._get_mask_path(index)
        mask = io.read_image_as_array(path)
        mask_patch = _utils.extract_mask_patch(mask, self, index)
        return tv_tensors.Mask(mask_patch, dtype=torch.int64)  # type: ignore[reportCallIssue]

    def _load_file_paths(self, split: Literal["train", "val", "test"] | None = None) -> List[str]:
        """Loads the file paths of the corresponding dataset split."""
        paths = sorted(glob.glob(os.path.join(self._root, "rgbs_colorNormalized/*.png")))
        if len(paths) != self._expected_length:
            raise ValueError(
                f"Expected {self._expected_length} images, found {len(paths)} in {self._root}."
            )

        train_indices, val_indices, test_indices = splitting.random_split(
            samples=paths,
            train_ratio=self._train_split_ratio,
            val_ratio=self._val_split_ratio,
            test_ratio=self._test_split_ratio,
            seed=self._seed,
        )

        match split:
            case "train":
                return [paths[i] for i in train_indices]
            case "val":
                return [paths[i] for i in val_indices]
            case "test":
                return [paths[i] for i in test_indices or []]
            case None:
                return paths
            case _:
                raise ValueError("Invalid split. Use 'train', 'val', 'test' or `None`.")

    def _get_mask_path(self, index):
        """Returns the path to the mask file corresponding to the patch at the given index."""
        return os.path.join(self._root, "masks", self.filename(index))
