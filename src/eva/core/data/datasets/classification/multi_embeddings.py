"""Dataset class for where a sample corresponds to multiple embeddings."""

import os
from typing import Callable, Dict, List, Literal

import numpy as np
import torch
from typing_extensions import override

from eva.core.data.datasets import embeddings as embeddings_base


class MultiEmbeddingsClassificationDataset(embeddings_base.EmbeddingsDataset[torch.Tensor]):
    """Dataset class for where a sample corresponds to multiple embeddings.

    Example use case: Slide level dataset where each slide has multiple patch embeddings.
    """

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: Literal["train", "val", "test"],
        column_mapping: Dict[str, str] = embeddings_base.default_column_mapping,
        embeddings_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ):
        """Initialize dataset.

        Expects a manifest file listing the paths of `.pt` files containing tensor embeddings.

        The manifest must have a `column_mapping["multi_id"]` column that contains the
        unique identifier group of embeddings. For oncology datasets, this would be usually
        the slide id. Each row in the manifest file points to a .pt file that can contain
        one or multiple embeddings (either as a list or stacked tensors). There can also be
        multiple rows for the same `multi_id`, in which case the embeddings from the different
        .pt files corresponding to that same `multi_id` will be stacked along the first dimension.

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

        self._multi_ids: List[int]

    @override
    def setup(self):
        super().setup()
        self._multi_ids = list(self._data[self._column_mapping["multi_id"]].unique())

    @override
    def load_embeddings(self, index: int) -> torch.Tensor:
        """Loads and stacks all embedding corresponding to the `index`'th multi_id."""
        # Get all embeddings for the given index (multi_id)
        multi_id = self._multi_ids[index]
        embedding_paths = self._data.loc[
            self._data[self._column_mapping["multi_id"]] == multi_id, self._column_mapping["path"]
        ].to_list()

        # Load embeddings and stack them accross the first dimension
        embeddings = []
        for path in embedding_paths:
            embedding = torch.load(os.path.join(self._root, path), map_location="cpu")
            if isinstance(embedding, list):
                embedding = torch.stack(embedding, dim=0)
            embeddings.append(embedding.unsqueeze(0) if embedding.ndim == 1 else embedding)
        embeddings = torch.cat(embeddings, dim=0)

        if not embeddings.ndim == 2:
            raise ValueError(f"Expected 2D tensor, got {embeddings.ndim} for {multi_id}.")

        return embeddings

    @override
    def load_target(self, index: int) -> np.ndarray:
        """Returns the target corresponding to the `index`'th multi_id.

        This method assumes that all the embeddings corresponding to the same `multi_id`
        have the same target. If this is not the case, it will raise an error.
        """
        multi_id = self._multi_ids[index]
        targets = self._data.loc[
            self._data[self._column_mapping["multi_id"]] == multi_id, self._column_mapping["target"]
        ]

        if not targets.nunique() == 1:
            raise ValueError(f"Multiple targets found for {multi_id}.")

        return np.asarray(targets.iloc[0], dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._multi_ids)
