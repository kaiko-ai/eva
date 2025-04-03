"""Balanced LiTS dataset."""

from typing import Callable, Dict, List, Literal, Tuple

import numpy as np
from typing_extensions import override

from eva.vision.data.datasets.segmentation import lits
from eva.vision.utils import io


class LiTSBalanced(lits.LiTS):
    """Balanced version of the LiTS - Liver Tumor Segmentation Challenge dataset.

    For each volume in the dataset, we sample the same number of slices where
    only the liver and where both liver and tumor are present.

    Webpage: https://competitions.codalab.org/competitions/17094

    For the splits we follow: https://arxiv.org/pdf/2010.01663v2
    """

    _expected_dataset_lengths: Dict[str | None, int] = {
        "train": 5514,
        "val": 1332,
        "test": 1530,
        None: 8376,
    }
    """Dataset version and split to the expected size."""

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
            seed: Seed used for generating the dataset splits and sampling of the slices.
        """
        super().__init__(root=root, split=split, transforms=transforms, seed=seed)

    @override
    def _create_indices(self) -> List[Tuple[int, int]]:
        """Builds the dataset indices for the specified split.

        Returns:
            A list of tuples, where the first value indicates the
            sample index which the second its corresponding slice
            index.
        """
        split_indices = set(self._get_split_indices())
        indices: List[Tuple[int, int]] = []
        random_generator = np.random.default_rng(seed=self._seed)

        for sample_idx in range(len(self._volume_files)):
            if sample_idx not in split_indices:
                continue

            segmentation_nii = io.read_nifti(self._segmentation_file(sample_idx))
            segmentation = io.nifti_to_array(segmentation_nii)
            tumor_filter = segmentation == 2
            tumor_slice_filter = tumor_filter.sum(axis=(0, 1)) > 0

            if tumor_filter.sum() == 0:
                continue

            liver_filter = segmentation == 1
            liver_slice_filter = liver_filter.sum(axis=(0, 1)) > 0

            liver_and_tumor_filter = liver_slice_filter & tumor_slice_filter
            liver_only_filter = liver_slice_filter & ~tumor_slice_filter

            n_slice_samples = min(liver_and_tumor_filter.sum(), liver_only_filter.sum())
            tumor_indices = list(np.where(liver_and_tumor_filter)[0])
            tumor_indices = list(
                random_generator.choice(tumor_indices, size=n_slice_samples, replace=False)
            )

            liver_indices = list(np.where(liver_only_filter)[0])
            liver_indices = list(
                random_generator.choice(liver_indices, size=n_slice_samples, replace=False)
            )

            indices.extend([(sample_idx, slice_idx) for slice_idx in tumor_indices + liver_indices])

        return list(indices)
