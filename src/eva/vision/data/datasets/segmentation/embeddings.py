"""Embeddings semantic segmentation dataset."""

import os
from typing import Callable, Dict, Literal

import pandas as pd
from typing_extensions import override

from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io

default_column_mapping: Dict[str, str] = {
    "path": "embeddings",
    "target": "target",
    "split": "split",
    "multi_id": "slide_id",
}
"""The default column mapping of the variables to the manifest columns."""


class EmbeddingsSegmentation(base.ImageSegmentation):
    """Embeddings semantic segmentation dataset."""

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: Literal["train", "val", "test"] | None = None,
        column_mapping: Dict[str, str] = default_column_mapping,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__(transforms=transforms)

        self._root = root
        self._manifest_file = manifest_file
        self._split = split
        self._column_mapping = default_column_mapping | column_mapping

    @override
    def configure(self) -> None:
        print("HERE")
        quit()

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
