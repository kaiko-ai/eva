"""China Consortium of Chest CT Image Investigation (CC-CCII) dataset."""

import os
from typing import Callable, Literal

import huggingface_hub
import numpy as np
import pandas as pd
import requests
import torch
from torchvision.datasets import utils as data_utils
from typing_extensions import override

from eva.vision.data.datasets import _validators
from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.tv_tensors import Volume


class CC_CCII(VisionDataset[Volume, torch.Tensor]):
    """China Consortium of Chest CT Image Investigation (CC-CCII) dataset.

    Dataset of the CT images and metadata are constructed from cohorts
    from the China Consortium of Chest CT Image Investigation (CC-CCII).
    All CT images are classified into novel coronavirus pneumonia (NCP)
    due to SARS-CoV-2 virus infection, common pneumonia and normal controls.
    This dataset is available globally with the aim to assist the clinicians
    and researchers to combat the COVID-19 pandemic.

    - Dataset description
      http://ncov-ai.big.ac.cn/download?lang=en
    """

    fold_id: Literal[0, 1, 2] = 0
    """The split-fold of the dataset."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] | None = None,
        download: bool = False,
        transforms: Callable | None = None,
        metadata_root: str | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the dataset root directory.
            split: Dataset split to use ('train', 'val' or `None`).
                If None, it uses the full dataset.
            download: Whether to download the dataset.
            transforms: A callable object for applying data transformations.
                If None, no transformations are applied.
            metadata_root: Root to the metadata. If `None`, it will
                use the root directory.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._download = download
        self._transforms = transforms
        self._metadata_root = metadata_root or root

        self._candidates: pd.DataFrame

    @override
    def prepare_data(self) -> None:
        if self._download:
            self._download_dataset()
            self._download_metadata()

    @override
    def configure(self) -> None:
        self._candidates = self._load_split_fold_metadata()

    @override
    def validate(self) -> None:
        def _valid_sample(index: int) -> bool:
            """Indicates if the sample files exist and are reachable."""
            ct_scan_path = self._get_ct_scan_file(index)
            return os.path.isfile(ct_scan_path)

        invalid_samples = [index for index in range(len(self)) if not _valid_sample(index)]
        if invalid_samples:
            raise OSError(
                f"Dataset '{self.__class__.__qualname__}' on root '{self._root}' contains "
                f"missing or corrupted samples  ({len(invalid_samples)} in total). "
                f"{_validators._SUFFIX_ERROR_MESSAGE} "
                f"Examples of missing folders: {str(invalid_samples[:10])[:-1]}, ...]. "
            )

    @override
    def filename(self, index):
        return self._get_ct_scan_file(index)

    @override
    def load_data(self, index: int) -> Volume:
        ct_scan_file = self._get_ct_scan_file(index)
        ct_scan_array = np.load(ct_scan_file)
        return Volume(ct_scan_array[:, None, :, :])

    @override
    def load_target(self, index: int) -> torch.Tensor:
        sample = self._candidates.loc[index]
        target = int(sample["target"])
        return torch.tensor(target, dtype=torch.long)

    @override
    def __len__(self) -> int:
        return len(self._candidates)

    def _load_split_fold_metadata(self) -> pd.DataFrame:
        split_fold_metadata_files = {
            "train": [f"CC_CCII_fold{self.fold_id}_train.csv"],
            "val": [f"CC_CCII_fold{self.fold_id}_valid.csv"],
            None: [
                f"CC_CCII_fold{self.fold_id}_train.csv",
                f"CC_CCII_fold{self.fold_id}_valid.csv",
            ],
        }
        split_fold_metadata = split_fold_metadata_files.get(self._split)
        if split_fold_metadata is None:
            raise ValueError("Invalid data split. Use 'train', 'val' or `None`.")
        return pd.concat(
            [
                pd.read_csv(os.path.join(self._metadata_root, filename))
                for filename in split_fold_metadata
            ]
        ).reset_index(drop=True)

    def _get_ct_scan_file(self, index: int) -> str:
        """Returns the CT scan file path from its index."""
        sample = self._candidates.loc[index]
        return os.path.join(
            self._root,
            "CC-CCII_public",
            "data",
            f"p{sample['patient_id']}-s{sample['scan_id']}.npy",
        )

    def _download_dataset(self) -> None:
        """Handles dataset download using Azure."""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Huggingface token required, please set the HF_TOKEN env variable.")

        huggingface_hub.snapshot_download(
            "Luffy503/VoCo_Downstream",
            repo_type="dataset",
            token=hf_token,
            local_dir=self._root,
            ignore_patterns=[".git*"],
            allow_patterns=["CC-CCII_public.zip"],
        )

        zip_path = os.path.join(self._root, "CC-CCII_public.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"CC-CCII_public.zip not found in {self._root}, "
                "something with the download went wrong."
            )

        data_utils.extract_archive(zip_path, self._root, remove_finished=True)

    def _download_metadata(self) -> None:
        """Efficiently downloads metadata files for dataset splits."""
        file_urls = [
            f"https://raw.githubusercontent.com/Luffy03/Large-Scale-Medical/refs/heads/main/Downstream/monai/CC-CCII/csv/CC_CCII_fold{fold}_{split}.csv"
            for fold in range(3)
            for split in ["train", "valid"]
        ]

        for url in file_urls:
            save_path = os.path.join(self._root, os.path.basename(url))
            with requests.get(url, stream=True, timeout=(5, 30)) as response:
                response.raise_for_status()
                with open(save_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
