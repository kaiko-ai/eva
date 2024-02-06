"""TotalSegmentator dataset class."""

import dataclasses
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets.utils import download_url, extract_archive
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.file_io import image_io


@dataclasses.dataclass(frozen=True)
class DownloadResource:
    """Contains download information for a specific resource."""

    filename: str
    url: str
    md5: str | None = None


@dataclasses.dataclass(frozen=True)
class SplitRatios:
    """Contains split ratios for train, val and test."""

    train: float = 0.6
    val: float = 0.2
    test: float = 0.2


class TotalSegmentator(VisionDataset[np.ndarray]):
    """TotalSegmentator dataset class.
    
    Create the multilabel classification dataset for the TotalSegmentator data.
    """

    resources: List[DownloadResource] = [
        DownloadResource(
            filename="Totalsegmentator_dataset_v201.zip",
            url="https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip",
            md5="6b5524af4b15e6ba06ef2d700c0c73e0",
        ),
    ]
    # TODO: switch to use complete instead of small dataset:
    #  - url="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"
    #  - md5="fe250e5718e0a3b5df4c4ea9d58a62fe"

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val", "test"],
        split_ratios: SplitRatios | None = None,
        sample_every_n_slice: int = 25,
        download: bool = False,
    ):
        """Initialize dataset.

        Args:
            root_dir: Path to the root directory of the dataset. The dataset will be downloaded
                and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.
            split_ratios: Ratios for the train, val and test splits.
            sample_every_n_slice: Number of slices to skip when sampling slices from the 3D images.
            download: Whether to download the data for the specified split.
                Note that the download will be executed only by additionally
                calling the :meth:`prepare_data` method and if the data does not exist yet on disk.
        """
        super().__init__()

        self._root_dir = root_dir
        self._split = split
        self._download = download

        self._data: pd.DataFrame
        self._path_key, self._split_key = "path", "split"
        self._split_ratios = split_ratios or self.default_split_ratios
        self._sample_every_n_slice = sample_every_n_slice
        self._manifest_path = os.path.join(self._root_dir, "manifest.csv")
        self._classes = None

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path, slice = self._get_image_path_and_slice(index)
        image = image_io.load_nifti_image_slice(image_path, slice)
        targets = np.asarray(self._data[self._classes].loc[index], dtype=np.int64)
        return image, targets

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
        if not os.path.isdir(os.path.join(self._root_dir, "Totalsegmentator_dataset_v201")):
            logger.info("Extracting archive ...")
            extract_archive(
                from_path=os.path.join(self._root_dir, "Totalsegmentator_dataset_v201.zip"),
                to_path=os.path.join(self._root_dir, "Totalsegmentator_dataset_v201")
            )
        self._classes = self._get_classes()

    @override
    def setup(self) -> None:
        df = self._load_dataset()
        if not os.path.isfile(self._manifest_path):
            self._save_manifest(df)
        df = self._generate_ordered_splits(df)
        self._verify_dataset(df)

        self._data = df.loc[df[self._split_key] == self._split].reset_index(drop=True)

    @property
    def default_split_ratios(self) -> SplitRatios:
        """Returns the default split ratios."""
        return SplitRatios()

    def _download_dataset(self) -> None:
        os.makedirs(self._root_dir, exist_ok=True)
        for r in self.resources:
            download_url(r.url, root=self._root_dir, filename=r.filename, md5=r.md5)

    def _get_image_path_and_slice(self, index: int) -> str:
        return (
            os.path.join(self._root_dir, self._data.at[index, self._path_key]),
            self._data.at[index, "slice"],
        )

    def _get_classes(self) -> List[str]:
        classes = [
            f.split(".")[0]
            for f in sorted(
                os.listdir(
                    os.path.join(
                        self._root_dir, "Totalsegmentator_dataset_v201/s0011/segmentations"
                    )
                )
            )
        ]
        return classes

    def _load_dataset(self) -> pd.DataFrame:
        """Loads the dataset manifest from a CSV file or creates the dataframe it does not exist."""

        if os.path.isfile(self._manifest_path):
            return pd.read_csv(self._manifest_path)
        
        data_dict = defaultdict(list)
        for i, path in enumerate(Path(self._root_dir).glob("**/*ct.nii.gz")):
            img_data = image_io.load_nifti_image(path)
            n_slices = img_data.shape[-1]

            # load all masks for an image:
            masks = {}
            for cl in self._classes:
                mask_path = os.path.join(path.parents[0], "segmentations", cl + ".nii.gz")
                masks[cl] = image_io.load_nifti_image(mask_path)

            # sample slices and extract label for each class:
            np.random.seed(i)
            start_slice = np.random.choice(min(self._sample_every_n_slice, n_slices))
            for i in range(start_slice, n_slices, self._sample_every_n_slice):
                data_dict["path"].append(path)
                data_dict["slice"].append(i)
                for cl in self._classes:
                    label = int(masks[cl][:, :, i].max())
                    data_dict[cl].append(label)

            df = pd.DataFrame(data_dict)

        return df
    
    def _save_manifest(self, df: pd.DataFrame) -> None:
        """Saves the dataset manifest to a CSV file."""

        manifest_path = os.path.join(self._root_dir, "manifest.csv")
        df.to_csv(manifest_path, index=False)

    def _generate_ordered_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orders each class by path and then splits it into train, val and test sets."""

        paths = sorted(df[self._path_key].unique())
        n_train_paths, n_val_paths = (
            round(len(paths) * self._split_ratios.train),
            round(len(paths) * self._split_ratios.val),
        )
        train_paths, val_paths, test_paths = (
            paths[:n_train_paths],
            paths[n_train_paths : n_train_paths + n_val_paths],
            paths[n_train_paths + n_val_paths :],
        )

        dfs = [
            pd.merge(df, pd.DataFrame({"path": train_paths, self._split_key: "train"}), on="path"),
            pd.merge(df, pd.DataFrame({"path": val_paths, self._split_key: "val"}), on="path"),
            pd.merge(df, pd.DataFrame({"path": test_paths, self._split_key: "test"}), on="path"),
        ]

        return pd.concat(dfs).reset_index(drop=True)

    def _verify_dataset(self, df: pd.DataFrame) -> None:
        if len(df) != 1454:
            raise ValueError(f"Expected 3633 samples but manifest lists {len(df)}.")
        
        if df.shape[1]-3 != len(self._classes) or len(self._classes) != 117:
            raise ValueError(f"Expected 117 classes but manifest lists {df.shape[1]-3}.")

        split_ratios = df["split"].value_counts(normalize=True)
        if not all(
            math.isclose(split_ratios[split], getattr(self._split_ratios, split), abs_tol=1e-2)
            for split in ["train", "val", "test"]
        ):
            raise ValueError(f"Unexpected split ratios: {split_ratios}.")
