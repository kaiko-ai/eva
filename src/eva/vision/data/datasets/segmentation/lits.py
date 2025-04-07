"""LiTS dataset."""

import functools
import glob
import os
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.core import utils
from eva.core.data import splitting
from eva.vision.data.datasets import _validators, vision
from eva.vision.utils import io


class LiTS(vision.VisionDataset[tv_tensors.Image, tv_tensors.Mask]):
    """LiTS - Liver Tumor Segmentation Challenge.

    Webpage: https://competitions.codalab.org/competitions/17094
    """

    _train_ratio: float = 0.7
    _val_ratio: float = 0.15
    _test_ratio: float = 0.15
    """Index ranges per split."""

    _fix_orientation: bool = True
    """Whether to fix the orientation of the images to match the default for radiologists."""

    _sample_every_n_slices: int | None = None
    """The amount of slices to sub-sample per 3D CT scan image."""

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 38686,
        "val": 11192,
        "test": 8760,
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
        seed: int = 8,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use.
            transforms: A function/transforms that takes in an image and a target
                mask and returns the transformed versions of both.
            seed: Seed used for generating the dataset splits.
        """
        super().__init__(transforms=transforms)

        self._root = root
        self._split = split
        self._seed = seed
        self._indices: List[Tuple[int, int]] = []

    @property
    @override
    def classes(self) -> List[str]:
        return ["background", "liver", "tumor"]

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
        for i in range(len(self._volume_files)):
            seg_path = self._segmentation_file(i)
            if not os.path.exists(seg_path):
                raise FileNotFoundError(
                    f"Segmentation file {seg_path} not found for volume {self._volume_files[i]}."
                )

        _validators.check_dataset_integrity(
            self,
            length=self._expected_dataset_lengths.get(self._split, 0),
            n_classes=3,
            first_and_last_labels=("background", "tumor"),
        )

    @override
    def load_data(self, index: int) -> tv_tensors.Image:
        sample_index, slice_index = self._indices[index]
        volume_path = self._volume_files[sample_index]
        image_nii = io.read_nifti(volume_path, slice_index)
        image_array = io.nifti_to_array(image_nii)
        if self._fix_orientation:
            image_array = self._orientation(image_array, sample_index)
        return tv_tensors.Image(image_array.transpose(2, 0, 1))

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        sample_index, slice_index = self._indices[index]
        segmentation_path = self._segmentation_file(sample_index)
        mask_nii = io.read_nifti(segmentation_path, slice_index)
        mask_array = io.nifti_to_array(mask_nii)
        if self._fix_orientation:
            semantic_labels = self._orientation(mask_array, sample_index)
        return tv_tensors.Mask(semantic_labels.squeeze(), dtype=torch.int64)  # type: ignore[reportCallIssue]

    def _orientation(self, array: npt.NDArray, sample_index: int) -> npt.NDArray:
        volume_path = self._volume_files[sample_index]
        orientation = io.fetch_nifti_axis_direction_code(volume_path)
        array = np.rot90(array, axes=(0, 1))
        if orientation == "LPS":
            array = np.flip(array, axis=0)
        return array.copy()

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

    def _segmentation_file(self, index: int) -> str:
        volume_file_path = self._volume_files[index]
        segmentation_file = os.path.basename(volume_file_path).replace("volume", "segmentation")
        return os.path.join(os.path.dirname(volume_file_path), segmentation_file)

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
        indices = list(range(len(self._volume_files)))
        train_indices, val_indices, test_indices = splitting.random_split(
            indices, self._train_ratio, self._val_ratio, self._test_ratio, seed=self._seed
        )
        split_indices_dict = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            None: indices,
        }
        if self._split not in split_indices_dict:
            raise ValueError("Invalid data split. Use 'train', 'val', 'test' or `None`.")
        return list(split_indices_dict[self._split])

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
