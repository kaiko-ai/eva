"""LiTS dataset."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.core import utils
from eva.vision.data.datasets import _utils as data_utils
from eva.vision.data.datasets import _validators
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io


class LiTS(base.ImageSegmentation):
    """LiTS - Liver Tumor Segmentation Challenge.

    Webpage: https://competitions.codalab.org/competitions/17094

    For the splits we follow: https://arxiv.org/pdf/2010.01663v2
    """

    _train_index_ranges: List[Tuple[int, int]] = [(0, 102)]
    _val_index_ranges: List[Tuple[int, int]] = [(102, 117)]
    _test_index_ranges: List[Tuple[int, int]] = [(117, 131)]
    """Index ranges per split."""

    _sample_every_n_slices: int | None = None
    """The amount of slices to sub-sample per 3D CT scan image."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 39307,
        "val": 12045,
        "test": 7286,
        None: 58638,
    }
    """Dataset version and split to the expected size."""

    _license: str = (
        "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License "
        "(https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)"
    )
    """Dataset license."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split

        self._indices: List[Tuple[int, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["liver", "tumor"]

    @functools.cached_property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @override
    def filename(self, index: int) -> str:
        sample_index, _ = self._indices[index]
        volume_file_path = self._volume_files[sample_index]
        return os.path.relpath(volume_file_path, self._root)

    @override
    def configure(self) -> None:
        self._indices = self._create_indices()

    @override
    def validate(self) -> None:
        if len(self._volume_files) != len(self._segmentation_files):
            raise ValueError(
                "The number of volume files does not match the number of the segmentation ones."
            )

        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split, 0),
            n_classes=2,
            first_and_last_labels=("liver", "tumor"),
        )

    @override
    def load_image(self, index: int) -> tv_tensors.Image:
        sample_index, slice_index = self._indices[index]
        volume_path = self._volume_files[sample_index]
        image_array = io.read_nifti(volume_path, slice_index)
        return tv_tensors.Image(image_array.transpose(2, 0, 1))

    @override
    def load_mask(self, index: int) -> tv_tensors.Mask:
        sample_index, slice_index = self._indices[index]
        segmentation_path = self._segmentation_files[sample_index]
        semantic_labels = io.read_nifti(segmentation_path, slice_index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        _, slice_index = self._indices[index]
        return {"slice_index": slice_index}

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _get_number_of_slices_per_volume(self, sample_index: int) -> int:
        """Returns the total amount of slices of a volume."""
        file_path = self._volume_files[sample_index]
        volume_shape = io.fetch_nifti_shape(file_path)
        return volume_shape[-1]

    @functools.cached_property
    def _volume_files(self) -> List[str]:
        files_pattern = os.path.join(self._root, "**", "volume-*.nii")
        files = glob.glob(files_pattern, recursive=True)
        return utils.numeric_sort(files)

    @functools.cached_property
    def _segmentation_files(self) -> List[str]:
        files_pattern = os.path.join(self._root, "**", "segmentation-*.nii")
        files = glob.glob(files_pattern, recursive=True)
        return utils.numeric_sort(files)

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
        """Returns the sample indices for the specified dataset split."""
        split_index_ranges = {
            "train": self._train_index_ranges,
            "val": self._val_index_ranges,
            "test": self._test_index_ranges,
            None: [(0, len(self._volume_files))],
        }
        index_ranges = split_index_ranges.get(self._split)
        if index_ranges is None:
            raise ValueError("Invalid data split. Use 'train', 'val', 'test' or `None`.")

        return data_utils.ranges_to_indices(index_ranges)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
