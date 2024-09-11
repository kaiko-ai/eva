"""Chest CT scan cancer classification dataset."""

import os
from typing import Callable, Dict, List, Literal, Tuple

import cv2
from torchvision.transforms.v2 import functional

import torch
from torchvision import tv_tensors
from torchvision.datasets import folder
from typing_extensions import override

from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class ChestCTScan(base.ImageClassification):
    """Chest CT scan cancer classification dataset.

    For more information and to download data see:
    https://tianchi.aliyun.com/dataset/93929
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
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

        self._samples: List[Tuple[str, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {
            "Adenocarcinoma": 0,
            "Large Cell Carcinoma": 1,
            "Normal": 2,
            "Squamous Cell Carcinoma": 3,
        }

    @override
    def configure(self) -> None:
        self._samples = self._make_dataset()

    @override
    def filename(self, index: int) -> str:
        image_path, _ = self._samples[index]
        return os.path.relpath(image_path, image_path)

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        image_path, _ = self._samples[index]
        image_array = io.read_image_as_array(image_path, flags=cv2.IMREAD_GRAYSCALE)
        return functional.to_image(image_array)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        _, target = self._samples[index]
        return torch.tensor(target, dtype=torch.long)

    @override
    def __len__(self) -> int:
        return len(self._samples)

    @property
    def _dataset_path(self) -> str:
        """Returns the path of the image data of the dataset."""
        split_directory = self._split if self._split in ["train", "test"] else "valid"
        return os.path.join(self._root, split_directory)

    def _make_dataset(self) -> List[Tuple[str, int]]:
        label_to_id = {
            "adenocarcinoma": 0,
            "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": 0,
            "large.cell.carcinoma": 1,
            "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": 1,
            "normal": 2,
            "squamous.cell.carcinoma": 3,
            "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": 3,
        }
        instances = []
        for root, _, fnames in sorted(os.walk(self._dataset_path, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if not folder.is_image_file(path):
                    continue

                target_class = os.path.basename(root)
                class_id = label_to_id[target_class]
                item = (path, class_id)
                instances.append(item)

        return instances
