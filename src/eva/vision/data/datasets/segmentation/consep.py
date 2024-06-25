"""CoNSeP segmentation dataset class."""

import glob
import os
from typing import Callable, Dict, List, Literal, Tuple

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import _validators
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class CoNSeP(base.ImageSegmentation):
    """CoNSeP segmentation dataset."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 27,
        "val": 14,
        None: 41,
    }
    """Dataset split to the expected size."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If `None`, the entire dataset is used.
            classes: Whether to configure the dataset with a subset of classes.
                If `None`, it will use all of them.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._samples_names: List[str] = []
        self._indices: List[Tuple[int, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return [
            "other",
            "inflammatory",
            "healthyepithelial",
            "dysplastic/malignant epithelial",
            "fibroblast",
            "muscle",
            "endothelial",
        ]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        return self._samples_names[index]

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

    @override
    def configure(self) -> None:
        self._samples_names = self._fetch_samples_names()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split),
            n_classes=len(self.classes),
            first_and_last_labels=((self.classes[0], self.classes[-1])),
        )

    @override
    def __len__(self) -> int:
        return len(self._samples_names)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        image_path = self._get_image_path(index)
        image = io.read_image_as_tensor(image_path)
        return tv_tensors.Image(image, dtype=torch.float32)

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        ground_truth = io.read_mat(self._get_mask_path(index))
        mask = ground_truth["type_map"]
        return tv_tensors.Mask(mask, dtype=torch.int64)

    def get_file_path(self, index: int, image_or_label: str, file_extension: str) -> str:
        """Returns the corresponding image path."""
        sample_name = self._samples_names[index]
        if self._split == "train":
            split = "Train"
        elif self._split == "val":
            split = "Test"
        else:
            if index <= self._expected_dataset_lengths["train"]:
                split = "Train"
            else:
                split = "Test"
        return os.path.join(self._root, split, image_or_label, sample_name + file_extension)

    def _get_image_path(self, sample_index: int) -> str:
        """Returns the corresponding image path."""
        return self.get_file_path(sample_index, "Images", ".png")

    def _get_mask_path(self, sample_index: int) -> str:
        """Returns the directory of the corresponding masks."""
        return self.get_file_path(sample_index, "Labels", ".mat")

    def _fetch_samples_names(self) -> List[str]:
        """Returns the name of all the samples of the selected dataset split."""

        def _samples_dirs_split(split: str) -> List[str]:
            split_dir = os.path.join(self._root, f"{split}/Images")
            return [os.path.basename(f).split(".")[0] for f in glob.glob(f"{split_dir}/*.png")]

        sample_filenames = []
        if self._split in ["train", None]:
            sample_filenames += _samples_dirs_split("Train")
        if self._split in ["val", None]:
            sample_filenames += _samples_dirs_split("Test")
        return sorted(sample_filenames)
