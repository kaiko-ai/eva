"""Bach dataset class."""

import dataclasses
import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets.utils import download_url, extract_archive
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.file_io import image_io


@dataclasses.dataclass
class DownloadResource:
    """Contains download information for a specific resource."""

    filename: str
    url: str
    md5: str | None = None


@dataclasses.dataclass
class SplitRatios:
    """Contains split ratios for train, val and test."""

    train: float = 0.6
    val: float = 0.1
    test: float = 0.3

    def __post_init__(self):
        """Verify that the split ratios sum to 1.0."""
        if self.train + self.val + self.test != 1.0:
            raise ValueError("Split ratios must sum to 1.0")


class BachDataset(VisionDataset[np.ndarray]):
    """Bach dataset class."""

    default_column_mapping: Dict[str, str] = {
        "path": "path",
        "target": "target",
        "split": "split",
    }

    classes: List[str] = [
        "Normal",
        "Benign",
        "InSitu",
        "Invasive",
    ]

    resources: List[DownloadResource] = [
        DownloadResource(
            filename="ICIAR2018_BACH_Challenge.zip",
            url="https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip",
            md5="8ae1801334aa943c44627c1eef3631b2",
        ),
    ]

    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        split: Literal["train", "val", "test"] | None,
        split_ratios: SplitRatios | None = None,
        download: bool = True,
        column_mapping: Dict[str, str] = default_column_mapping,
    ):
        """Initialize dataset.

        Args:
            root_dir: Path to the root directory of the dataset. The dataset will be downloaded
                and extracted here, if it does not already exist.
            manifest_path: Path to the dataset manifest file.
            split: Dataset split to use. If None, the entire dataset is used.
            split_ratios: Ratios for the train, val and test splits.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not exist yet on disk.
            column_mapping: Mapping between the standardized column names and the actual
                column names in the provided manifest file.
        """
        super().__init__()

        self._root_dir = root_dir
        self._manifest_path = manifest_path
        self._split = split
        self._download = download
        self._column_mapping = column_mapping

        self._data: pd.DataFrame

        self._path_column = self._column_mapping["path"]
        self._split_column = self._column_mapping["split"]
        self._target_column = self._column_mapping["target"]

        if split_ratios is None:
            self._split_ratios = SplitRatios(train=0.7, val=0.15, test=0.15)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = image_io.load_image(self._get_image_path(index))
        target = self._data.at[index, self._target_column]
        return image, target

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        if not os.path.isdir(os.path.join(self._root_dir, "ICIAR2018_BACH_Challenge")):
            logger.info("Extracting archive ...")
            extract_archive(
                from_path=os.path.join(self._root_dir, "ICIAR2018_BACH_Challenge.zip"),
                to_path=self._root_dir,
            )
        if not os.path.exists(self._manifest_path):
            self._create_manifest()

    @override
    def setup(self) -> None:
        self._data = self._load_manifest()

        if self._split:
            split_filter = self._data[self._split_column] == self._split
            self._data = self._data.loc[split_filter].reset_index(drop=True)

    @property
    def _class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _download_dataset(self) -> None:
        os.makedirs(self._root_dir, exist_ok=True)

        for r in self.resources:
            download_url(r.url, root=self._root_dir, filename=r.filename, md5=r.md5)

    def _get_image_path(self, index: int) -> str:
        return os.path.join(self._root_dir, self._data.at[index, self._path_column])

    def _load_manifest(self) -> pd.DataFrame:
        logger.info(f"Load manifest from {self._manifest_path}.")
        df_manifest = pd.read_parquet(self._manifest_path)
        self._verify_manifest(df_manifest)
        return df_manifest

    def _verify_manifest(self, df_manifest: pd.DataFrame) -> None:
        if len(df_manifest) != 400:
            raise ValueError(f"Expected 400 samples but manifest lists {len(df_manifest)}.")

        if not (df_manifest["target"].value_counts() == 100).all():
            raise ValueError("Expected 100 samples per class.")

    def _create_manifest(self) -> pd.DataFrame:
        # load image paths & targets
        df_manifest = pd.DataFrame({self._path_column: Path(self._root_dir).glob("**/*.tif")})
        df_manifest[self._target_column] = df_manifest[self._path_column].apply(
            lambda p: Path(p).parent.name
        )

        if not all(df_manifest[self._target_column].isin(self.classes)):
            raise ValueError(f"Unexpected classes: {df_manifest[self._target_column].unique()}")

        df_manifest[self._target_column] = df_manifest[self._target_column].map(self._class_to_idx)  # type: ignore
        df_manifest[self._path_column] = df_manifest[self._path_column].apply(
            lambda x: Path(x).relative_to(self._root_dir).as_posix()
        )

        # create splits
        # TODO: refactor this into a shared split module: https://github.com/kaiko-ai/eva/issues/75
        df_manifest[self._split_column] = ""
        dfs = []
        for _, df_target in df_manifest.groupby(self._target_column):
            df_target = df_target.sort_values(by=self._path_column).reset_index(drop=True)
            n_train, n_val = round(df_target.shape[0] * self._split_ratios.train), round(
                df_target.shape[0] * self._split_ratios.val
            )
            df_target.loc[:n_train, self._split_column] = "train"
            df_target.loc[n_train : n_train + n_val, self._split_column] = "val"
            df_target.loc[n_train + n_val :, self._split_column] = "test"
            dfs.append(df_target)

        # save manifest
        df_manifest = pd.concat(dfs).reset_index(drop=True)
        self._verify_manifest(df_manifest)
        df_manifest.to_parquet(self._manifest_path)
        logger.info(f"Saved manifest to {self._manifest_path}")

        return df_manifest
