"""Embeddings classification dataset."""

import os

import torch
from typing_extensions import override

from eva.core.data.datasets import embeddings as embeddings_base


class EmbeddingsClassificationDataset(embeddings_base.EmbeddingsDataset[torch.Tensor]):
    """Embeddings dataset class for classification tasks."""

    @override
    def load_embeddings(self, index: int) -> torch.Tensor:
        filename = self.filename(index)
        embeddings_path = os.path.join(self._root, filename)
        tensor = torch.load(embeddings_path, map_location="cpu")
        if isinstance(tensor, list):
            if len(tensor) > 1:
                raise ValueError(
                    f"Expected a single tensor in the .pt file, but found {len(tensor)}."
                )
            tensor = tensor[0]
        return tensor.squeeze(0)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        target = self._data.at[index, self._column_mapping["target"]]
        return torch.tensor(target, dtype=torch.int64)

    @override
    def __len__(self) -> int:
        return len(self._data)
