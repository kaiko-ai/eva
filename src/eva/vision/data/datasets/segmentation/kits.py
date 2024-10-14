"""KiTS23 dataset."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchvision import tv_tensors
from urllib import request
from typing_extensions import override
from eva.core.utils.progress_bar import tqdm

from eva.core import utils
from eva.core.data import splitting
from eva.vision.data.datasets import _utils, _validators, structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class KiTS23(base.ImageSegmentation):
    """KiTS23 - The 2023 Kidney and Kidney Tumor Segmentation challenge.

    Webpage: https://kits-challenge.org/kits23/
    """

    _train_index_ranges: List[Tuple[int, int]] = [(0, 300), (400, 589)]
    """Train range indices."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 38686,
        "test": 8760,
    }
    """Dataset version and split to the expected size."""

    _sample_every_n_slices: int | None = None
    """The amount of slices to sub-sample per 3D CT scan image."""

    _license: str = "CC BY-NC-SA 4.0"
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train"],
        download: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does
                not yet exist on disk.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._download = download

        self._indices: List[Tuple[int, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["kidney", "tumor", "cyst"]

    @functools.cached_property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        sample_index, _ = self._indices[index]
        return self._volume_filename(sample_index)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def configure(self) -> None:
        self._indices = self._create_indices()

    @override
    def validate(self) -> None:
        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split, 0),
            n_classes=3,
            first_and_last_labels=("kidney", "cyst"),
        )

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        sample_index, slice_index = self._indices[index]
        volume_path = self._volume_path(sample_index)
        image_array = io.read_nifti(volume_path, slice_index)
        return tv_tensors.Image(image_array.transpose(2, 0, 1))

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        sample_index, slice_index = self._indices[index]
        segmentation_path = self._segmentation_path(sample_index)
        semantic_labels = io.read_nifti(segmentation_path, slice_index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        _, slice_index = self._indices[index]
        return {"slice_index": slice_index}

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _create_indices(self) -> List[Tuple[int, int]]:
        """Builds the dataset indices for the specified split.

        Returns:
            A list of tuples, where the first value indicates the
            sample index which the second its corresponding slice
            index.
        """
        indices = [
            (sample_idx, slide_idx)
            for sample_idx in self._get_split_indices()
            for slide_idx in range(self._get_number_of_slices_per_volume(sample_idx))
            if slide_idx % (self._sample_every_n_slices or 1) == 0
        ]
        return indices

    def _get_split_indices(self) -> List[int]:
        """Builds the dataset indices for the specified split."""
        split_index_ranges = {
            "train": self._train_index_ranges,
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train' or `test`.")

        return _utils.ranges_to_indices(index_ranges)

    def _get_number_of_slices_per_volume(self, sample_index: int) -> int:
        """Returns the total amount of slices of a volume."""
        volume_shape = io.fetch_nifti_shape(self._volume_path(sample_index))
        return volume_shape[-1]

    def _volume_filename(self, sample_index: int) -> str:
        return os.path.join(f"case_{sample_index}", "imaging.nii.gz")

    def _segmentation_filename(self, sample_index: int) -> str:
        return os.path.join(f"case_{sample_index}", "segmentation.nii.gz")

    def _volume_path(self, sample_index: int) -> str:
        return os.path.join(self._root, self._volume_filename(sample_index))

    def _segmentation_path(self, sample_index: int) -> str:
        return os.path.join(self._root, self._segmentation_filename(sample_index))

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        self._print_license()
        for case_id in tqdm(
            self._get_split_indices(),
            desc=">> Downloading dataset",
            leave=False,
        ):
            image_path, segmentation_path = self._volume_path(case_id), self._segmentation_path(case_id)
            if os.path.isfile(image_path) and os.path.isfile(segmentation_path):
                continue

            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            request.urlretrieve(
                url=f"https://kits19.sfo2.digitaloceanspaces.com/master_{case_id:05d}.nii.gz",
                filename=image_path,
            )
            request.urlretrieve(
                url=f"https://github.com/neheller/kits23/raw/refs/heads/main/dataset/case_{case_id:05d}/segmentation.nii.gz",
                filename=segmentation_path,
            )

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
