"""TotalSegmentator 2D segmentation dataset class."""

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Literal, Tuple

import cv2
from glob import glob
import numpy as np
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class TotalSegmentator2D(base.ImageSegmentation):
    """TotalSegmentator 2D segmentation dataset."""

    resources_full: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_v201.zip",
            url="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
            md5="fe250e5718e0a3b5df4c4ea9d58a62fe",
        ),
    ]
    """Complete dataset resources."""

    resources_small: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_small_v201.zip",
            url="https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip",
            md5="6b5524af4b15e6ba06ef2d700c0c73e0",
        ),
    ]
    """Resources for the small dataset version."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None,
        version: Literal["small", "full"] = "small",
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
        image_target_transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
            image_target_transforms: A function/transforms that takes in an
                image and a label and returns the transformed versions of both.
                This transform happens after the `image_transforms` and
                `target_transforms`.
        """
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
            image_target_transforms=image_target_transforms,
        )

        self._root = root
        self._split = split
        self._version = version
        self._download = download

        self._samples: List[str] = []
        self._indices: List[int] = []

    @property
    def classes(self) -> List[str] | None:
        sample_targets = os.path.join(
            self._root, random.choice(os.listdir(self._root)), "segmentations"
        )
        classes = [file.split(".")[0] for file in os.listdir(sample_targets)]
        return sorted(classes)

    @property
    def class_to_idx(self) -> None:
        return {index: label for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        sample_name = self._sample_dir(index)
        return os.path.join(sample_name, "ct.nii.gz")

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def setup(self) -> None:
        self._samples = os.listdir(self._root)
        self._indices = list(range(len(self._samples)))

    @override
    def load_image_and_mask(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_3D = self._load_image(index)
        image_2D, slice_idx = self._extract_image_slice(image_3D)
        mask = self._load_mask(index, slice_idx)
        return image_2D, mask

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _sample_dir(self, index: int) -> str:
        return self._samples[self._indices[index]]

    def _load_image(self, index: int) -> np.ndarray:
        sample_dir = self._sample_dir(index)
        image_path = os.path.join(self._root, sample_dir, "ct.nii.gz")
        return io.read_nifti(image_path)

    def _extract_image_slice(self, image_3D: np.ndarray) -> np.ndarray:
        slice_idx = random.randrange(image_3D.shape[2])
        image_rgb = cv2.cvtColor(image_3D[:, :, slice_idx], cv2.COLOR_GRAY2RGB)
        return image_rgb, slice_idx

    def _load_mask(self, index: int, slice_index: int = 0) -> np.ndarray:
        sample_dir = self._sample_dir(index)
        masks_dir = os.path.join(self._root, sample_dir, "segmentations", "*.nii.gz")
        return np.stack([io.read_nifti(path, slice_index) for path in sorted(glob(masks_dir))])

    def _download_dataset(self) -> None:
        dataset_resources = {
            "small": self.resources_small,
            "full": self.resources_full,
        }
        resources = dataset_resources.get(self._split)
        if resources is None:
            raise ValueError("Invalid data split. Use 'small' or 'full'.")

        for resource in resources:
            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )
