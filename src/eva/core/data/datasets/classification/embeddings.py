"""Embeddings classification dataset."""

import os
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from typing_extensions import override

from eva.core.data.datasets import base
from eva.core.utils import io


class EmbeddingsClassificationDataset(base.Dataset):
    """Embeddings classification dataset."""

    default_column_mapping: Dict[str, str] = {
        "data": "embeddings",
        "target": "target",
        "split": "split",
    }
    """The default column mapping of the variables to the manifest columns."""

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: str | None = None,
        column_mapping: Dict[str, str] = default_column_mapping,
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
        super().__init__()

        self._root = root
        self._manifest_file = manifest_file
        self._split = split
        self._column_mapping = self.default_column_mapping | column_mapping
        self._embeddings_transforms = embeddings_transforms
        self._target_transforms = target_transforms

        self._data: pd.DataFrame

    def filename(self, index: int) -> str:
        """Returns the filename of the `index`'th data sample.

        Note that this is the relative file path to the root.

        Args:
            index: The index of the data-sample to select.

        Returns:
            The filename of the `index`'th data sample.
        """
        return self._data.at[index, self._column_mapping["data"]]

    @override
    def setup(self):
        self._data = self._load_manifest()

    def __getitem__(self, index) -> Tuple[torch.Tensor, np.ndarray]:
        """Returns the `index`'th data sample.

        Args:
            index: The index of the data-sample to select.

        Returns:
            A data sample and its target.
        """
        embeddings = self._load_embeddings(index)
        target = self._load_target(index)
        return self._apply_transforms(embeddings, target)

    def __len__(self) -> int:
        """Returns the total length of the data."""
        return len(self._data)

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

    def _load_target(self, index: int) -> np.ndarray:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample target as an array.
        """
        target = self._data.at[index, self._column_mapping["target"]]
        return np.asarray(target, dtype=np.int64)

    def _load_manifest(self) -> pd.DataFrame:
        """Loads manifest file and filters the data based on the split column.

        Returns:
            The data as a pandas DataFrame.
        """
        manifest_path = os.path.join(self._root, self._manifest_file)
        data = io.read_dataframe(manifest_path)
        if self._split is not None:
            filtered_data = data.loc[data[self._column_mapping["split"]] == self._split]
            data = filtered_data.reset_index(drop=True)
        return data

    def _apply_transforms(
        self, embeddings: torch.Tensor, target: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Applies the transforms to the provided data and returns them.

        Args:
            embeddings: The embeddings to be transformed.
            target: The training target.

        Returns:
            A tuple with the embeddings and the target transformed.
        """
        if self._embeddings_transforms is not None:
            embeddings = self._embeddings_transforms(embeddings)

        if self._target_transforms is not None:
            target = self._target_transforms(target)

        return embeddings, target
