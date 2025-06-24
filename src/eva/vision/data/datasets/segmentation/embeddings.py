"""Embeddings based semantic segmentation dataset."""

import os
from typing import List

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.core.data.datasets import embeddings as embeddings_base


class EmbeddingsSegmentationDataset(embeddings_base.EmbeddingsDataset[tv_tensors.Mask]):
    """Embeddings segmentation dataset."""

    @override
    def load_embeddings(self, index: int) -> List[torch.Tensor]:
        filename = self.filename(index)
        embeddings_path = os.path.join(self._root, filename)
        embeddings = torch.load(embeddings_path, map_location="cpu")
        if isinstance(embeddings, torch.Tensor):
            embeddings = [embeddings]
        return [tensor.squeeze(0) for tensor in embeddings]

    @override
    def load_target(self, index: int) -> tv_tensors.Mask:
        filename = self._data.at[index, self._column_mapping["target"]]
        mask_path = os.path.join(self._root, filename)
        semantic_labels = torch.load(mask_path, map_location="cpu")
        return tv_tensors.Mask(semantic_labels, dtype=torch.int64)  # type: ignore[reportCallIssue]

    @override
    def __len__(self) -> int:
        return len(self._data)
