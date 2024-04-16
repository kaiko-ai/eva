import bisect
import os
from typing import Any, Dict

import numpy as np

from eva.core.models.modules.typings import DATA_SAMPLE
from eva.vision.data.datasets.wsi import MultiWsiDataset


class MultiWsiClassificationDataset(MultiWsiDataset):
    def __getitem__(self, index: int) -> DATA_SAMPLE:
        data = super().__getitem__(index)
        target = self._load_target(index)
        metadata = self._load_metadata(index)

        return DATA_SAMPLE(data, target, metadata)

    # TODO: create panda-specific dataset class for functions below
    def _load_target(self, index: int) -> np.ndarray:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return self._manifest.at[dataset_idx, self._column_mapping["target"]]

    def _load_metadata(self, index: int) -> Dict[str, Any]:
        return {"slide_id": self.filename(index).split(".")[0]}
    
    def filename(self, index: int) -> str:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        full_path = self._manifest.at[dataset_idx, self._column_mapping["path"]]
        return os.path.basename(full_path)
