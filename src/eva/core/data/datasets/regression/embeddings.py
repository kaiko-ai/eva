"""Embeddings regression dataset."""

import torch
from typing_extensions import override

from eva.core.data.datasets.classification import EmbeddingsClassificationDataset


class EmbeddingsRegressionDataset(EmbeddingsClassificationDataset):
    """Embeddings dataset class for regression tasks."""

    @override
    def load_target(self, index: int) -> torch.Tensor:
        target = self._data.at[index, self._column_mapping["target"]]
        return torch.tensor(float(target), dtype=torch.float32)
