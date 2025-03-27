"""Base dataset class for Embeddings."""

import abc
import multiprocessing
import os
from typing import Callable, Dict, Generic, Literal, Tuple, TypeVar

import pandas as pd
import torch
from typing_extensions import override

from eva.core.data.datasets import base
from eva.core.utils import io

TargetType = TypeVar("TargetType")
"""The target data type."""


default_column_mapping: Dict[str, str] = {
    "path": "embeddings",
    "target": "target",
    "split": "split",
    "multi_id": "wsi_id",
}
"""The default column mapping of the variables to the manifest columns."""


class EmbeddingsDataset(base.Dataset, Generic[TargetType]):
    """Abstract base class for embedding datasets."""

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: Literal["train", "val", "test"] | None = None,
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
        self._column_mapping = default_column_mapping | column_mapping
        self._embeddings_transforms = embeddings_transforms
        self._target_transforms = target_transforms

        self._data: pd.DataFrame

        self._set_multiprocessing_start_method()

    def filename(self, index: int) -> str:
        """Returns the filename of the `index`'th data sample.

        Note that this is the relative file path to the root.

        Args:
            index: The index of the data-sample to select.

        Returns:
            The filename of the `index`'th data sample.
        """
        return self._data.at[index, self._column_mapping["path"]]

    @override
    def setup(self):
        self._data = self._load_manifest()

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the total length of the data."""

    def __getitem__(self, index) -> Tuple[torch.Tensor, TargetType]:
        """Returns the `index`'th data sample.

        Args:
            index: The index of the data-sample to select.

        Returns:
            A data sample and its target.
        """
        embeddings = self.load_embeddings(index)
        target = self.load_target(index)
        return self._apply_transforms(embeddings, target)

    @abc.abstractmethod
    def load_embeddings(self, index: int) -> torch.Tensor:
        """Returns the `index`'th embedding sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The embedding sample as a tensor.
        """

    @abc.abstractmethod
    def load_target(self, index: int) -> TargetType:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample target as an array.
        """

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
        self, embeddings: torch.Tensor, target: TargetType
    ) -> Tuple[torch.Tensor, TargetType]:
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

    def _set_multiprocessing_start_method(self):
        """Sets the multiprocessing start method to spawn.

        If the start method is not set explicitly, the torch data loaders will
        use the OS default method, which for some unix systems is `fork` and
        can lead to runtime issues such as deadlocks in this context.
        """
        multiprocessing.set_start_method("spawn", force=True)
