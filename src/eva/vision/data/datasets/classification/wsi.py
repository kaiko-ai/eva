"""Dataset classes for whole-slide image classification."""

import bisect
import os
from typing import Any, Dict

import torch
from typing_extensions import override

from eva.core.models.modules.typings import DATA_SAMPLE
from eva.vision.data.datasets.wsi import MultiWsiDataset


class MultiWsiClassificationDataset(MultiWsiDataset):
    """Classification Dataset class for reading patches from multiple whole-slide images.

    # TODO: Replace this by dataset specific classes?
    """

    @override
    def filename(self, index: int) -> str:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        full_path = self._manifest.at[dataset_idx, self._column_mapping["path"]]
        return os.path.basename(full_path)

    @override
    def __getitem__(self, index: int) -> DATA_SAMPLE:
        data = super().__getitem__(index)
        target = self._load_target(index)
        metadata = self._load_metadata(index)

        return DATA_SAMPLE(data, target, metadata)

    def _load_target(self, index: int) -> torch.Tensor:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        target = self._manifest.at[dataset_idx, self._column_mapping["target"]]
        return torch.tensor(target)

    def _load_metadata(self, index: int) -> Dict[str, Any]:
        return {"slide_id": self.filename(index).split(".")[0]}
