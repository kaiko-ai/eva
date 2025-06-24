"""BRACS dataset class."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import torch
from torchvision import tv_tensors
from torchvision.datasets import folder
from typing_extensions import override

from eva.vision.data.datasets import _validators, vision
from eva.vision.utils import io


class BRACS(vision.VisionDataset[tv_tensors.Image, torch.Tensor]):
    """Dataset class for BRACS images and corresponding targets."""

    _expected_dataset_lengths: Dict[str, int] = {
        "train": 3657,
        "val": 312,
        "test": 570,
    }
    """Expected dataset lengths for the splits and complete dataset."""

    _license: str = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the root directory of the dataset.
            split: Dataset split to use.
            transforms: A function/transform which returns a transformed
                version of the raw data samples.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split

        self._samples: List[Tuple[str, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        image_path, *_ = self._samples[index]
        return os.path.relpath(image_path, self._dataset_path)

    @override
    def prepare_data(self) -> None:
        _validators.check_dataset_exists(self._root, True)

    @override
    def configure(self) -> None:
        self._samples = self._make_dataset()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths[self._split],
            n_classes=7,
            first_and_last_labels=("0_N", "6_IC"),
        )

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        image_path, _ = self._samples[index]
        return io.read_image_as_tensor(image_path)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        _, target = self._samples[index]
        return torch.tensor(target, dtype=torch.long)

    @override
    def __len__(self) -> int:
        return len(self._samples)

    @property
    def _dataset_path(self) -> str:
        """Returns the full path of dataset directory."""
        return os.path.join(self._root, "BRACS_RoI/latest_version")

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Builds the dataset for the specified split."""
        dataset = folder.make_dataset(
            directory=os.path.join(self._dataset_path, self._split),
            class_to_idx=self.class_to_idx,
            extensions=(".png"),
        )
        return dataset

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
