"""MHIST dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision
from eva.vision.utils import io


class MHIST(vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """MHIST dataset."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        transforms: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Path to the root directory of the dataset.
            split: Dataset split to use.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        super().__init__(transforms=transforms)

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
    def load_data(self, index: int) -> tv_tensors.Image:
        image_filename, _ = self._samples[index]
        image_path = os.path.join(self._dataset_path, image_filename)
        return io.read_image_as_tensor(image_path)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        _, label = self._samples[index]
        target = self.class_to_idx[label]
        return torch.tensor(target, dtype=torch.float32)

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
