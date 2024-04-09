import bisect
import random
from typing import Any, Dict

import numpy as np

from eva.core.models.modules.typings import INPUT_BATCH
from eva.vision.data.datasets.wsi import MultiWsiDataset


class MultiWsiClassificationDataset(MultiWsiDataset):
    def __getitem__(self, index: int) -> INPUT_BATCH:
        data = super().__getitem__(index)
        target = self._load_target(index)
        metadata = self._load_metadata(index)

        return INPUT_BATCH(data, target, metadata)

    def _load_target(self, index: int) -> np.ndarray:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return self._manifest.at[dataset_idx, self._column_mapping["target"]]

    def _load_metadata(self, index: int) -> Dict[str, Any]:
        # TODO: Implement metadata loading
        return {"slide_id": random.randint(0, 100)}
