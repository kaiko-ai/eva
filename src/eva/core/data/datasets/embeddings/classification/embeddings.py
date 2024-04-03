"""Embeddings classification dataset."""

import os
from typing import Callable, Dict

import numpy as np
import torch
from typing_extensions import override

from eva.core.data.datasets.embeddings import base


class EmbeddingsClassificationDataset(base.EmbeddingsDataset):
    """Embeddings classification dataset."""

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: str | None = None,
        column_mapping: Dict[str, str] = base.default_column_mapping,
        embeddings_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Expects a manifest file listing the paths of .pt files that contain
        tensor embeddings of shape [embedding_dim] or [1, embedding_dim].

        Args:
            root: Root directory of the dataset.
            manifest_file: The path to the manifest file, which is relative to
                the `root` argument.
            split: The dataset split to use. The `split` column of the manifest
                file will be splitted based on this value.
            column_mapping: Defines the map between the variables and the manifest
                columns. It will overwrite the `default_column_mapping` with
                the provided values, so that `column_mapping` can contain only the
                values which are altered or missing.
            embeddings_transforms: A function/transform that transforms the embedding.
            target_transforms: A function/transform that transforms the target.
        """
        super().__init__(
            root=root,
            manifest_file=manifest_file,
            split=split,
            column_mapping=column_mapping,
            embeddings_transforms=embeddings_transforms,
            target_transforms=target_transforms,
        )

    @override
    def _load_embeddings(self, index: int) -> torch.Tensor:
        """Returns the `index`'th embedding sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample embedding as an array.
        """
        filename = self.filename(index)
        embeddings_path = os.path.join(self._root, filename)
        tensor = torch.load(embeddings_path, map_location="cpu")
        return tensor.squeeze(0)

    @override
    def _load_target(self, index: int) -> np.ndarray:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample target as an array.
        """
        target = self._data.at[index, self._column_mapping["target"]]
        return np.asarray(target, dtype=np.int64)

    def __len__(self) -> int:
        """Returns the total length of the data."""
        return len(self._data)
