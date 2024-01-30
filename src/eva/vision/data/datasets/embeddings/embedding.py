"""Embedding dataset."""

import os
from typing import Dict

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from typing_extensions import override

from eva.vision.data.datasets.typings import DatasetType
from eva.vision.data.datasets.vision import VisionDataset


class EmbeddingDataset(VisionDataset):
    """Embedding dataset."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "slide_id": "slide_id",
        "mask": "mask",
    }

    def __init__(
        self,
        manifest_path: str,
        root_dir: str,
        split: str | None,
        column_mapping: Dict[str, str] = default_column_mapping,
        dataset_type: DatasetType = DatasetType.PATCH,
        n_patches_per_slide: int = 1000,
        seed: int = 42,
    ):
        """Initialize dataset.

        Expects a manifest file listing the paths of .pt files that contain the embeddings.
        There are two supported shapes for the embedding tensors & files:
        a. Patch Tasks: A single .pt file per patch containing a tensor of shape [embedding_dim]
            or [1, embedding_dim]
        b. Slide Tasks: Each slide can have either one or multiple .pt files, each containing
            a sequence of patch embeddings of shape [k, embedding_dim].

        Args:
            manifest_path: Path to the manifest file. Can be either a .csv or .parquet file.
            root_dir: Root directory of the dataset. If specified, the paths in the manifest
                file are expected to be relative to this directory.
            split: Dataset split to use. If None, the entire dataset is used.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
            dataset_type: Type of the dataset (DatasetType.SLIDE or DatasetType.PATCH level).
            n_patches_per_slide: Number of patches to sample per slide. Only used if dataset_type
                is set to DatasetType.SLIDE.
            seed: Seed used for sampling patches when dataset_type is DatasetType.SLIDE.
        """
        super().__init__()

        self._manifest_path = manifest_path
        self._root_dir = root_dir
        self._split = split
        self._column_mapping = column_mapping
        self._dataset_type = dataset_type
        self._n_patches_per_slide = n_patches_per_slide
        self._seed = seed

        self._data: pd.DataFrame

        self._path_column = self._column_mapping["path"]
        self._slide_id_column = self._column_mapping["slide_id"]
        self._mask_column = self._column_mapping["mask"]
        self._embedding_column = "embedding"

    @override
    def __getitem__(self, index) -> torch.Tensor:
        return self._data.at[index, self._embedding_column]

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def setup(self):
        self._data = self._load_manifest()
        self._data[self._embedding_column] = None

        for index in tqdm.tqdm(self._data.index):
            self._data.at[index, self._embedding_column] = self._load_embedding_file(index)

        if self._dataset_type == DatasetType.SLIDE:
            self._data = self._sample_n_patches_per_slide(self._data)

        self._data = self._data.reset_index(drop=True)

    def _load_embedding_file(self, index) -> torch.Tensor:
        path = self._get_embedding_path(index)
        tensor = torch.load(path, map_location="cpu")
        orig_shape = tensor.shape
        if self._dataset_type == DatasetType.PATCH:
            tensor = tensor.squeeze(0)
            if tensor.ndim != 1:
                raise ValueError(f"Unexpected tensor shape {orig_shape} for {path}")
        elif self._dataset_type == DatasetType.SLIDE:
            if tensor.ndim != 2:
                raise ValueError(f"Unexpected tensor shape {orig_shape} for {path}")
        return tensor

    def _get_embedding_path(self, index: int) -> str:
        return os.path.join(self._root_dir, self._data.at[index, self._path_column])

    def _load_manifest(self) -> pd.DataFrame:
        if self._manifest_path.endswith(".csv"):
            return pd.read_csv(self._manifest_path)
        elif self._manifest_path.endswith(".parquet"):
            return pd.read_parquet(self._manifest_path)
        else:
            raise ValueError(f"Unsupported file format for manifest file {self._manifest_path}")

    def _sample_n_patches_per_slide(self, df: pd.DataFrame) -> pd.DataFrame:
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
            n_masked = 0
            if pad_size and dim0 < pad_size:
                n_masked = pad_size - dim0
                stacked_embeddings = F.pad(
                    stacked_embeddings, pad=(0, 0, 0, n_masked), mode="constant", value=0
                )
                mask = torch.zeros(pad_size, 1).bool()
                mask[-n_masked:, :] = True
            else:
                mask = torch.zeros(stacked_embeddings.shape[0], 1).bool()

            result = {self._embedding_column: stacked_embeddings, self._mask_column: mask}
            metdata_columns = set(df.columns) - set(result.keys())
            for col in metdata_columns:
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

        return df
