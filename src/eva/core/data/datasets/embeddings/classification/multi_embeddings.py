"""Dataset class for slide embeddings (composed of multiple patch embeddings)."""

import os
from typing import Callable, Dict, List, Literal

import numpy as np
import torch
from typing_extensions import override

from eva.core.data.datasets.embeddings import base


class MultiEmbeddingsClassificationDataset(base.EmbeddingsDataset):
    """Embedding dataset."""

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: Literal["train", "val", "test"],
        column_mapping: Dict[str, str] = base.default_column_mapping,
        embeddings_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ):
        """Initialize dataset.

        Expects a manifest file listing the paths of `.pt` files. Each slide can have either
        one or multiple `.pt` files, each containing a sequence of patch embeddings of shape
        `[k, embedding_dim]`.

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
            manifest_file=manifest_file,
            root=root,
            split=split,
            column_mapping=column_mapping,
            embeddings_transforms=embeddings_transforms,
            target_transforms=target_transforms,
        )

        self._slide_ids: List[int]

    @override
    def setup(self):
        super().setup()
        self._slide_ids = list(self._data[self._column_mapping["slide_id"]].unique())

    @override
    def _load_embeddings(self, index: int) -> torch.Tensor:
        """Returns the `index`'th embedding sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample embedding as an array.
        """
        # Get all embeddings for the slide
        slide_id = self._slide_ids[index]
        embedding_paths = self._data.loc[
            self._data[self._column_mapping["slide_id"]] == slide_id, self._column_mapping["path"]
        ].to_list()
        embedding_paths = [os.path.join(self._root, path) for path in embedding_paths]

        # Load embeddings and stack
        embeddings = [torch.load(path, map_location="cpu") for path in embedding_paths]
        embeddings = torch.cat(embeddings, dim=0)

        if not embeddings.ndim == 2:
            raise ValueError(f"Expected 2D tensor, got {embeddings.ndim} for slide {slide_id}.")

        return embeddings

    @override
    def _load_target(self, index: int) -> np.ndarray:
        """Returns the `index`'th target sample.

        Args:
            index: The index of the data sample to load.

        Returns:
            The sample target as an array.
        """
        slide_id = self._slide_ids[index]
        slide_targets = self._data.loc[
            self._data[self._column_mapping["slide_id"]] == slide_id, self._column_mapping["target"]
        ]

        if not slide_targets.nunique() == 1:
            raise ValueError(f"Multiple targets found for slide {slide_id}.")

        return slide_targets.iloc[0]

    def __len__(self) -> int:
        """Returns the total length of the data."""
        return len(self._data)
