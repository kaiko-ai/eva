"""Bach dataset class."""
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets.utils import download_and_extract_archive
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset


class BachDataset(VisionDataset[np.ndarray]):
    """Bach dataset class."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
        "split": "split",
    }

    classes = [
        "Normal",
        "Benign",
        "InSitu",
        "Invasive",
    ]

    resources = {
        "ICIAR2018_BACH_Challenge.zip": "https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip",
    }

    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        split: str | None,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        """Initialize dataset.

        Args:
            root_dir: Path to the root directory of the dataset. The dataset will be downloaded
                and extracted here, if it does not already exist.
            manifest_path: Path to the dataset manifest file.
            split: Dataset split to use. If None, the entire dataset is used.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
        """
        super().__init__()

        self._root_dir = root_dir
        self._manifest_path = manifest_path
        self._split = split
        self._column_mapping = column_mapping

        self._data: pd.DataFrame

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _exists(self) -> bool:
        return all(
            os.path.join(self._root_dir, p) for p in self._data[self._column_mapping["path"]]
        )

    def _download(self) -> None:
        os.makedirs(self._root_dir, exist_ok=True)

        for filename, url in self.resources.items():
            download_and_extract_archive(url, download_root=self._root_dir, filename=filename)

    @override
    def _prepare_data(self) -> None:
        if not self._exists():
            self._download()

        self._create_manifest()

    @override
    def _setup(self) -> None:
        self._data = self._load_manifest()

        if self._split:
            self._data = self._data[self._data[self._column_mapping["split"]] == self._split]

    @override
    def __len__(self) -> int:
        return len(self._data)

    def _load_manifest(self) -> pd.DataFrame:
        logger.info(f"Load manifest from {self._manifest_path}")
        return pd.read_parquet(self._manifest_path)

    def _create_manifest(self) -> pd.DataFrame:
        # load image paths & labels
        df_manifest = pd.DataFrame(Path(self._root_dir).glob("**/*.tif"), columns=["path"])
        df_manifest["label"] = df_manifest["path"].apply(lambda p: Path(p).parent.name)

        if not df_manifest["label"].isin(self.classes).all():
            raise ValueError(f"Unexpected classes: {df_manifest['label'].unique()}")

        # create splits
        train_fraction, val_fraction = 0.7, 0.15
        df_manifest["split"] = ""
        dfs = []
        for label in df_manifest["label"].unique():
            label_filter = df_manifest["label"] == label
            df = df_manifest[label_filter].sort_values(by="path").reset_index(drop=True)
            n_train, n_val = round(df.shape[0] * train_fraction), round(df.shape[0] * val_fraction)
            df.loc[:n_train, "split"] = "train"
            df.loc[n_train : n_train + n_val, "split"] = "val"
            df.loc[n_train + n_val :, "split"] = "test"
            dfs.append(df)

        # save manifest
        df_manifest = pd.concat(dfs).sort_values(by=["split", "label"]).reset_index(drop=True)
        df_manifest.to_parquet(self._manifest_path)
        logger.info(f"Saved manifest to {self._manifest_path}")
