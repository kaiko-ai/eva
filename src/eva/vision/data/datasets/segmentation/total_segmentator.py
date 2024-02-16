"""TotalSegmentator dataset class."""

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Literal, Tuple

from glob import glob
import numpy as np
import pandas as pd
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import structs
from eva.vision.data.datasets.classification import base
from eva.vision.utils import io


class TotalSegmentator(base.ImageClassification):
    """TotalSegmentator dataset."""

    resources: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_v201.zip",
            url="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
            md5="fe250e5718e0a3b5df4c4ea9d58a62fe",
        ),
    ]
    """Complete dataset resources."""

    resources_small: List[structs.DownloadResource] = [
        structs.DownloadResource(
            filename="Totalsegmentator_dataset_small_v201.zip",
            url="https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip",
            md5="6b5524af4b15e6ba06ef2d700c0c73e0",
        ),
    ]
    """Resources for the small dataset version."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None,
        sample_every_n_slices: int = 25,
        small_version: bool = True,
        download: bool = False,
        image_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root: Path to the root directory of the dataset. The dataset will
                be downloaded and extracted here, if it does not already exist.
            split: Dataset split to use. If None, the entire dataset is used.

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
        self._small_version = small_version
        self._sample_every_n_slices = sample_every_n_slices
        self._download = download

        self._samples: List[str] = []
        self._indices: List[int] = []

    @property
    def classes(self) -> List[str] | None:
        sample_targets = os.path.join(
            self._root, random.choice(os.listdir(self._root)), "segmentations"
        )
        classes = [file.split(".")[0] for file in os.listdir(sample_targets)]
        return sorted(classes)

    @property
    def class_to_idx(self) -> None:
        return {index: label for index, label in enumerate(self.classes)}

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()

    @override
    def setup(self) -> None:
        self._samples = os.listdir(self._root)
        self._indices = list(range(len(self._samples)))

    @override
    def load_image(self, index: int) -> np.ndarray:
        sample_name = self._samples[self._indices[index]]
        image_path = os.path.join(self._root, sample_name, "ct.nii.gz")
        return io.read_nifti(image_path, slice_index=0)

    @override
    def load_mask(self, index: int) -> np.ndarray:
        sample_name = self._samples[self._indices[index]]
        masks_dir = os.path.join(self._root, sample_name, "segmentations", "*.nii.gz")
        masks = np.stack([io.read_nifti(path, 0) for path in sorted(glob(masks_dir))])
        return masks

    @override
    def load_target(self, index: int) -> np.ndarray:
        targets = self._data[self._classes].loc[index]
        return np.asarray(targets, dtype=np.int64)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def filename(self, index: int) -> str:
        return self._data.at[index, self._path_key]

    def _download_dataset(self) -> None:
        resources = self.resources_small if self._small_version else self.resources
        for resource in resources:
            utils.download_and_extract_archive(
                resource.url,
                download_root=self._root,
                filename=resource.filename,
                remove_finished=True,
            )

    def _get_image_path_and_slice(self, index: int) -> Tuple[str, int]:
        return (
            os.path.join(self._root, self._data.at[index, self._path_key]),
            self._data.at[index, "slice"],
        )

    def _get_classes(self) -> List[str]:
        classes = [
            f.split(".")[0]
            for f in sorted(
                os.listdir(
                    os.path.join(self._root, "Totalsegmentator_dataset_v201/s0011/segmentations")
                )
            )
        ]
        if len(classes) == 0:
            raise ValueError("No classes found in the dataset.")

        return classes

    def _load_dataset(self) -> pd.DataFrame:
        """Loads the dataset manifest from a CSV file or creates the dataframe it does not exist."""
        if os.path.isfile(self._manifest_path):
            return pd.read_csv(self._manifest_path)

        data_dict = defaultdict(list)
        for i, path in enumerate(Path(self._root).glob("**/*ct.nii.gz")):
            img_data = io.read_nifti(str(path))
            n_slices = img_data.shape[-1]

            # load all masks for an image:
            masks = {}
            for cl in self._classes:
                mask_path = os.path.join(path.parents[0], "segmentations", cl + ".nii.gz")
                masks[cl] = io.read_nifti(mask_path)

            # sample slices and extract label for each class:
            np.random.seed(i)
            start_slice = np.random.choice(min(self._sample_every_n_slices, n_slices))
            for i in range(start_slice, n_slices, self._sample_every_n_slices):
                data_dict["path"].append(path)
                data_dict["slice"].append(i)
                for cl in self._classes:
                    label = int(masks[cl][:, :, i].max())
                    data_dict[cl].append(label)

            df = pd.DataFrame(data_dict)

        return df  # type: ignore

    def _save_manifest(self, df: pd.DataFrame) -> None:
        """Saves the dataset manifest to a CSV file."""
        manifest_path = os.path.join(self._root, "manifest.csv")
        df.to_csv(manifest_path, index=False)

    def _load_manifest(self) -> pd.DataFrame:
        """Loads the dataset manifest from a CSV file."""
        manifest_path = os.path.join(self._root, "manifest.csv")
        return pd.read_csv(manifest_path)

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

        if df.shape[1] - 3 != len(self._classes) or len(self._classes) != 117:
            raise ValueError(f"Expected 117 classes but manifest lists {df.shape[1]-3}.")

        split_ratios = df["split"].value_counts(normalize=True)
        if not all(
            math.isclose(split_ratios[split], getattr(self._split_ratios, split), abs_tol=1e-2)
            for split in ["train", "val", "test"]
        ):
            raise ValueError(f"Unexpected split ratios: {split_ratios}.")
