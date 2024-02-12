"""Bach dataset class."""

import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal

import numpy as np
import pandas as pd
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class Bach(base.ImageClassification):
    """Bach dataset class."""

    classes: List[str] = [
        "Normal",
        "Benign",
        "InSitu",
        "Invasive",
    ]

    resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="ICIAR2018_BACH_Challenge.zip",
            url="https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip",
            md5="8ae1801334aa943c44627c1eef3631b2",
        ),
    ]

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None,
        split_ratios: structs.SplitRatios | None = None,
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ):
        """Initialize the dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
            split_ratios: Ratios for the train, val and test splits.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not
                exist yet on disk.
            image_transforms: A function/transform that takes in an image
                and returns a transformed version.
            target_transforms: A function/transform that takes in the target
                and transforms it.
        """
        super().__init__(
            image_transforms=image_transforms,
            target_transforms=target_transforms,
        )

        self._root = root
        self._split = split
        self._download = download

        self._data: pd.DataFrame
        self._path_key, self._split_key, self._target_key = "path", "split", "target"
        self._split_ratios = split_ratios or self.default_split_ratios

    @property
    def default_split_ratios(self) -> structs.SplitRatios:
        """Returns the defaults split ratios."""
        return structs.SplitRatios(train=0.6, val=0.1, test=0.3)

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def setup(self) -> None:
        df = self._load_dataset()
        df = self._generate_ordered_stratified_splits(df)
        self._verify_dataset(df)

        self._data = df.loc[df[self._split_key] == self._split].reset_index(drop=True)

    @override
    def load_image(self, index: int) -> np.ndarray:
        filename = os.path.join(self._root, self._data.at[index, self._path_key])
        return io.read_image(filename)

    @override
    def load_target(self, index: int) -> np.ndarray:
        target = self._data.at[index, self._target_key]
        return np.asarray(target, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._data)

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        for resource in self.resources:
            file_path = os.path.join(self._root, resource.filename)
            if utils.check_integrity(file_path, resource.md5):
                continue

            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=False,
            )

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.DataFrame({self._path_key: Path(self._root).glob("**/*.tif")})
        df[self._target_key] = df[self._path_key].apply(lambda p: Path(p).parent.name)

        if not all(df[self._target_key].isin(self.classes)):
            raise ValueError(f"Unexpected classes: {df[self._target_key].unique()}")

        df[self._target_key] = df[self._target_key].map(self._class_to_idx)  # type: ignore
        df[self._path_key] = df[self._path_key].apply(
            lambda x: Path(x).relative_to(self._root).as_posix()
        )
        return df

    @property
    def _class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _generate_ordered_stratified_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orders each class by path and then splits it into train, val and test sets."""
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
