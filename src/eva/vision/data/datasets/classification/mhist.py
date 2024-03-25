"""MHIST dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from typing_extensions import override

from eva.vision.data.datasets import _validators
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class MHIST(base.ImageClassification):
    """MHIST dataset."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Path to the root directory of the dataset.
            split: Dataset split to use.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
        """
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
        )

        self._root = root
        self._split = split

        self._samples: List[Tuple[str, str]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["SSA", "HP"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"SSA": 0, "HP": 1}

    @override
    def filename(self, index: int) -> str:
        image_filename, _ = self._samples[index]
        return image_filename

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, False)

    @override
    def configure(self) -> None:
        self._samples = self._make_dataset()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=2175 if self._split == "train" else 977,
            n_classes=2,
            first_and_last_labels=("SSA", "HP"),
        )

    @override
    def load_image(self, index: int) -> np.ndarray:
        image_filename, _ = self._samples[index]
        image_path = os.path.join(self._dataset_path, image_filename)
        return io.read_image(image_path)

    @override
    def load_target(self, index: int) -> np.ndarray:
        _, label = self._samples[index]
        target = self.class_to_idx[label]
        return np.asarray(target, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._samples)

    def _make_dataset(self) -> List[Tuple[str, str]]:
        """Generates and returns a list of samples of a form (image_filename, label)."""
        data = io.read_csv(self._annotations_path)
        samples = [
            (sample["Image Name"], sample["Majority Vote Label"])
            for sample in data
            if sample["Partition"] == self._split
        ]
        return samples

    @property
    def _dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        return os.path.join(self._root, "images")

    @property
    def _annotations_path(self) -> str:
        """Returns the path of the annotations file of the dataset."""
        return os.path.join(self._root, "annotations.csv")
