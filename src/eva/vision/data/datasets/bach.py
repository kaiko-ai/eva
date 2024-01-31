"""Bach dataset class."""

import dataclasses
import math
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


class BachDataset(VisionDataset[np.ndarray]):
    """Bach dataset class."""

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
        split: Literal["train", "val", "test"],
        split_ratios: SplitRatios | None = None,
        download: bool = True,
    ):
        """Initialize dataset.

        Args:
            root_dir: Path to the root directory of the dataset. The dataset will be downloaded
                and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
            split_ratios: Ratios for the train, val and test splits.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not exist yet on disk.
        """
        super().__init__()

        self._root_dir = root_dir
        self._split = split
        self._download = download

        self._data: pd.DataFrame
        self._path_key, self._split_key, self._target_key = "path", "split", "target"

        if split_ratios is None:
            self._split_ratios = SplitRatios(train=0.7, val=0.15, test=0.15)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image = image_io.load_image(self._get_image_path(index))
        target = self._data.at[index, self._target_key]
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

    @override
    def setup(self) -> None:
        df = self._load_dataset()
        df = self._generate_splits(df)
        self._verify_dataset(df)

        self._data = df.loc[df[self._split_key] == self._split].reset_index(drop=True)

    @property
    def _class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _download_dataset(self) -> None:
        os.makedirs(self._root_dir, exist_ok=True)
        for r in self.resources:
            download_url(r.url, root=self._root_dir, filename=r.filename, md5=r.md5)

    def _get_image_path(self, index: int) -> str:
        return os.path.join(self._root_dir, self._data.at[index, self._path_key])

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame({self._path_key: Path(self._root_dir).glob("**/*.tif")})
        df[self._target_key] = df[self._path_key].apply(lambda p: Path(p).parent.name)

        if not all(df[self._target_key].isin(self.classes)):
            raise ValueError(f"Unexpected classes: {df[self._target_key].unique()}")

        df[self._target_key] = df[self._target_key].map(self._class_to_idx)  # type: ignore
        df[self._path_key] = df[self._path_key].apply(
            lambda x: Path(x).relative_to(self._root_dir).as_posix()
        )

        return df

    def _generate_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: refactor this into a shared spliting module: https://github.com/kaiko-ai/eva/issues/75
        df[self._split_key] = ""
        dfs = []
        for _, df_target in df.groupby(self._target_key):
            df_target = df_target.sort_values(by=self._path_key).reset_index(drop=True)
            n_train, n_val = round(df_target.shape[0] * self._split_ratios.train), round(
                df_target.shape[0] * self._split_ratios.val
            )
            df_target.loc[:n_train, self._split_key] = "train"
            df_target.loc[n_train : n_train + n_val, self._split_key] = "val"
            df_target.loc[n_train + n_val :, self._split_key] = "test"
            dfs.append(df_target)

        df = pd.concat(dfs).reset_index(drop=True)

        return df

    def _verify_dataset(self, df: pd.DataFrame) -> None:
        if len(df) != 400:
            raise ValueError(f"Expected 400 samples but manifest lists {len(df)}.")

        if not (df["target"].value_counts() == 100).all():
            raise ValueError("Expected 100 samples per class.")

        split_ratios = df["split"].value_counts(normalize=True)
        if not all(
            math.isclose(split_ratios[split], getattr(self._split_ratios, split), abs_tol=1e-5)
            for split in ["train", "val", "test"]
        ):
            raise ValueError(f"Unexpected split ratios: {split_ratios}.")
