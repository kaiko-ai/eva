"""Dataset class for slide embeddings (composed of multiple patch embeddings)."""

from typing import Any, Dict, Literal, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from typing_extensions import override

from eva.vision.data.datasets.embeddings.patch import PatchEmbeddingDataset


class SlideEmbeddingDataset(PatchEmbeddingDataset):
    """Embedding dataset."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
        "split": "split",
        "slide_id": "slide_id",
    }

    def __init__(
        self,
        manifest_path: str,
        root: str,
        split: Literal["train", "val", "test"],
        column_mapping: Dict[str, str] = default_column_mapping,
        n_patches_per_slide: int = 1000,
        pad_value: int | float = float("-inf"),
        seed: int = 42,
    ):
        """Initialize dataset.

        Expects a manifest file listing the paths of .pt files. Each slide can have either
        one or multiple .pt files, each containing a sequence of patch embeddings of shape
        [k, embedding_dim]. This dataset class will then stack the patch embeddings for each
        slide, sample n_patches_per_slide patch embeddings (or pad with zeros if there are less
        patches) and finally return tensors of shape [n_patches_per_slide, embedding_dim] suitable
        to train multi-instance learning (MIL) heads. For slides with less than n_patches_per_slide
        patches, the resulting tensor is padded.

        Args:
            manifest_path: Path to the manifest file. Can be either a .csv or .parquet file, with
                the required columns: path, target, split, slide_id (names can be adjusted
                using the column_mapping parameter).
            root: Root directory of the dataset. If specified, the paths in the manifest
                file are expected to be relative to this directory.
            split: Dataset split to use.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
            n_patches_per_slide: Number of patches to sample per slide.
            pad_value: Value used for padding the embeddings of slides with less than
                n_patches_per_slide patches.
            seed: Seed used for sampling patches per slide.
        """
        super().__init__(
            manifest_path=manifest_path,
            root=root,
            split=split,
            column_mapping=column_mapping,
        )

        self._n_patches_per_slide = n_patches_per_slide
        self._seed = seed
        self._pad_value = pad_value
        self._slide_id_column = self._column_mapping["slide_id"]

    @override
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        metadata = {"slide_id": self._data.at[index, self._slide_id_column]}
        return (
            self._data.at[index, self._embedding_column],
            self._data.at[index, self._target_column],
            metadata,
        )

    @override
    def setup(self):
        super().setup()
        self._data = self._sample_n_patches_per_slide(self._data)

    @override
    def _load_embedding_file(self, index) -> torch.Tensor:
        path = self._get_embedding_path(index)
        tensor = torch.load(path, map_location="cpu")
        if tensor.ndim != 2:
            import pytest

            pytest.set_trace()
            raise ValueError(f"Unexpected tensor shape {tensor.shape} for {path}")
        return tensor

    def _sample_n_patches_per_slide(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function randomly selects n_patches_per_slide patch embeddings per slide.

        After sampling, the patch embeddings of each slide are stacked as tensors of shape
        [n_patches_per_slide, embedding_dim]. For slides that have less than n_patches_per_slide
        patches, the resulting tensor is padded with self._pad_value.
        """
        if self._embedding_column not in df.columns:
            raise ValueError(f"Column {self._embedding_column} not found in dataframe.")
        if self._slide_id_column not in df.columns:
            raise ValueError(f"Column {self._slide_id_column} not found in dataframe.")

        def try_sample(df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
            """Samples n_samples from the dataframe if it contains at least n_samples entries."""
            if len(df) < n_samples:
                return df
            else:
                return df.sample(n=n_samples, random_state=seed)

        def stack_and_pad(df: pd.DataFrame, pad_size: int) -> pd.Series:
            """Stacks the embeddings of the patches of a slide and pads the resulting tensor."""
            stacked_embeddings = torch.cat(df[self._embedding_column].tolist(), dim=0)

            dim0 = stacked_embeddings.shape[0]
            if pad_size and dim0 < pad_size:
                stacked_embeddings = F.pad(
                    stacked_embeddings,
                    pad=(0, 0, 0, pad_size - dim0),
                    mode="constant",
                    value=self._pad_value,
                )

            result = {self._embedding_column: stacked_embeddings}
            remaining_columns = set(df.columns) - set(result.keys())
            for col in remaining_columns:
                # metadata is the same for patches of the same slide, so just take the first entry
                result[col] = df[col].iloc[0]

            return pd.Series(result)

        # transform dataframe such that each row contains a single patch embedding
        df[self._embedding_column] = df[self._embedding_column].apply(
            lambda e: list(torch.split(e, 1, 0))
        )
        df = df.explode(self._embedding_column)

        # sample patches
        df = (
            df.groupby(self._slide_id_column)
            .apply(try_sample, n_samples=self._n_patches_per_slide, seed=self._seed)
            .reset_index(drop=True)
        )

        # transform dataframe such that each row represents a single slide with an embedding tensor
        # of shape [_n_patches_per_slide, embedding_dim] (slides with less patches are padded)
        df = df.groupby(self._slide_id_column).apply(
            stack_and_pad, pad_size=self._n_patches_per_slide
        )

        return df.reset_index(drop=True)
