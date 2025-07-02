"""PubMedQA dataset class."""

import os
import random
from typing import Dict, List, Literal

import torch
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from typing_extensions import override

from eva.language.data.datasets.classification import base


class PubMedQA(base.TextClassification):
    """Dataset class for PubMedQA question answering task."""

    _license: str = "MIT License (https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE)"
    """Dataset license."""

    def __init__(
        self,
        root: str | None = None,
        split: Literal["train", "val", "test"] | None = None,
        download: bool = False,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the PubMedQA dataset.

        Args:
            root: Directory to cache the dataset. If None, no local caching is used.
            split: Valid splits among ["train", "val", "test"].
                If None, it will use "train+test+validation".
            download: Whether to download the dataset if not found locally. Default is False.
            max_samples: Maximum number of samples to use. If None, use all samples.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._download = download
        self._max_samples = max_samples

    def _load_dataset(self, dataset_path: str | None) -> Dataset:
        """Loads the PubMedQA dataset from the local cache or downloads it.

        Args:
            dataset_path: The path to the local cache (may be None).

        Returns:
            The loaded dataset object.
        """
        dataset_name = "bigbio/pubmed_qa"
        config_name = "pubmed_qa_labeled_fold0_source"
        split = (self._split or "train+test+validation") if self._split != "val" else "validation"

        if self._download:
            logger.info("Downloading dataset from HuggingFace Hub")
            raw_dataset = load_dataset(
                dataset_name,
                name=config_name,
                split=split,
                trust_remote_code=True,
                download_mode="reuse_dataset_if_exists",
            )
            if dataset_path:
                raw_dataset.save_to_disk(dataset_path)  # type: ignore
                logger.info(f"Dataset saved to: {dataset_path}")
        else:
            if not dataset_path or not os.path.exists(dataset_path):
                raise ValueError(
                    "Dataset path not found. Set download=True or provide a valid root path."
                )

            logger.info(f"Loading dataset from: {dataset_path}")
            raw_dataset = load_from_disk(dataset_path)

        return raw_dataset  # type: ignore

    @override
    def prepare_data(self) -> None:
        """Downloads and prepares the PubMedQA dataset.

        If `self._root` is None, the dataset is used directly from HuggingFace.
        Otherwise, it checks if the dataset is already cached in `self._root`.
        If not cached, it downloads the dataset into `self._root`.
        """
        dataset_path = None

        if self._root:
            dataset_path = self._root
            os.makedirs(self._root, exist_ok=True)

        try:
            self.dataset = self._load_dataset(dataset_path)
            if self._max_samples is not None and len(self.dataset) > self._max_samples:
                logger.info(
                    f"Subsampling dataset from {len(self.dataset)} to {self._max_samples} samples"
                )
                random.seed(42)
                indices = random.sample(range(len(self.dataset)), self._max_samples)
                self.dataset = self.dataset.select(indices)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare dataset: {e}") from e

    @property
    @override
    def classes(self) -> List[str]:
        return ["no", "yes", "maybe"]

    @property
    @override
    def class_to_idx(self) -> Dict[str, int]:
        return {"no": 0, "yes": 1, "maybe": 2}

    @override
    def load_text(self, index: int) -> str:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        sample = dict(self.dataset[index])
        return f"Question: {sample['QUESTION']}\nContext: " + " ".join(sample["CONTEXTS"])

    @override
    def load_target(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        return torch.tensor(
            self.class_to_idx[self.dataset[index]["final_decision"]], dtype=torch.long
        )

    @override
    def load_metadata(self, index: int) -> Dict[str, str]:
        sample = self.dataset[index]
        return {
            "year": sample.get("YEAR") or "",
            "labels": sample.get("LABELS") or "",
            "meshes": sample.get("MESHES") or "",
            "long_answer": sample.get("LONG_ANSWER") or "",
            "reasoning_required": sample.get("reasoning_required_pred") or "",
            "reasoning_free": sample.get("reasoning_free_pred") or "",
        }

    @override
    def __len__(self) -> int:
        return len(self.dataset)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
