"""TotalSegmentor datasets."""

import functools
import os
from glob import glob
from typing import List

from typing_extensions import override

from eva.vision.data.datasets.segmentation.total_segmentator import base


class TotalSegmentator2D(base.TotalSegmentator2DBase):
    """TotalSegmentator Segmentation: Whole body CT segmentation dataset.

    References:
      - https://github.com/wasserth/TotalSegmentator
    """

    @functools.cached_property
    @override
    def classes(self) -> List[str]:
        first_sample_labels = os.path.join(
            self._root, self._samples_dirs[0], "segmentations", "*.nii.gz"
        )
        return sorted(map(_get_filename, glob(first_sample_labels)))


def _get_filename(path: str) -> str:
    """Returns the filename from the full path."""
    return os.path.basename(path).split(".")[0]
