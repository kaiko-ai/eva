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

default_column_mapping: Dict[str, str] = {
    "path": "path",
    "target": "target",
    "slide_id": "slide_id",
}


class EmbeddingDataset(VisionDataset):
    """Embedding dataset."""

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

        Args:
            manifest_path: Path to the manifest file.
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
        self._target_column = self._column_mapping["target"]
        self._embedding_column = self._column_mapping["embedding"]
        self._slide_id_column = self._column_mapping["slide_id"]

    @override
    def __getitem__(self, index) -> torch.Tensor:
        return self._data.at[index, self._embedding_column]

    @override
    def setup(self):
        self._data = self._load_manifest()
        self._data[self._embedding_column] = None

        for index, _ in tqdm.tqdm(self._data):
            self._data.at[index, self._embedding_column] = self._load_embedding_file(index)

        if self._dataset_type == DatasetType.SLIDE:
            self._data = self._sample_n_patches_per_slide(self._data)

    def _load_embedding_file(self, index) -> torch.Tensor:
        return torch.load(self._get_embedding_path(index), map_location="cpu")

    def _get_embedding_path(self, index: int) -> str:
        return os.path.join(self._root_dir, self._data.at[index, self._path_column])

    def _load_manifest(self) -> pd.DataFrame:
        return pd.read_parquet(self._manifest_path)

    def _sample_n_patches_per_slide(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._embedding_column not in df.columns:
            raise ValueError(f"Column {self._embedding_column} not found in dataframe.")
        if self._slide_id_column not in df.columns:
            raise ValueError(f"Column {self._slide_id_column} not found in dataframe.")

        def try_sample(df: pd.DataFrame, n_samples: int, seed: int):
            if len(df) < n_samples:
                return df
            else:
                return df.sample(n=n_samples, random_state=seed)

        def stack_and_pad(df: pd.DataFrame, pad_size: int):
            stacked_embeddings = torch.cat(df[self._embedding_column].tolist(), dim=0)

            dim0 = stacked_embeddings.shape[0]
            n_masked = 0
            if pad_size and dim0 < pad_size:
                # [n_patches_in_slide, embedding_dim] -> [pad_size, embedding_dim]
                n_masked = pad_size - dim0
                stacked_embeddings = F.pad(
                    stacked_embeddings, pad=(0, 0, 0, n_masked), mode="constant", value=0
                )
                mask = torch.zeros(pad_size, 1).bool()
                mask[-n_masked:, :] = True
            else:
                mask = torch.zeros(stacked_embeddings.shape[0], 1).bool()

            # all the patches belong to the same slide and therefore have the same label
            target = df[self._target_column].iloc[0] if self._target_column in df.columns else None

            return pd.Series(
                {
                    self._target_column: target,
                    self._embedding_column: stacked_embeddings,
                    "mask": mask,
                }
            )

        df[self._embedding_column] = df[self._embedding_column].apply(
            lambda e: list(torch.split(e, 1, 0))
        )
        df = df.explode(self._embedding_column)

        df = (
            df.groupby(self._slide_id_column)
            .apply(try_sample, n_samples=self._n_patches_per_slide, seed=self._seed)
            .reset_index(drop=True)
        )

        df = df.groupby(self._slide_id_column).apply(
            stack_and_pad, pad_size=self._n_patches_per_slide
        )

        return df
