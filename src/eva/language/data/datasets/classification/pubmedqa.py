"""PubMedQA dataset class."""

import os
from typing import Dict, List, Literal

import torch
from datasets import Dataset, load_dataset
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
    ) -> None:
        """Initialize the PubMedQA dataset.

        Args:
            root: Directory to cache the dataset. If None, no local caching is used.
            split: Valid splits among ["train", "val", "test"].
                If None, it will use "train+test+validation".
            download: Whether to download the dataset if not found locally. Default is False.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._download = download

    def _load_dataset(self, dataset_cache_path: str | None) -> Dataset:
        """Loads the PubMedQA dataset from the local cache or downloads it if needed.

        Args:
            dataset_cache_path: The path to the local cache (may be None).

        Returns:
            The loaded Dataset object.
        """
        if dataset_cache_path is not None and os.path.exists(dataset_cache_path):
            dataset_path = dataset_cache_path
            logger.info(f"Loaded dataset from local cache: {dataset_cache_path}")
            is_local = True
        else:
            if not self._download and self._root:
                raise ValueError(
                    "Dataset not found locally and downloading is disabled. "
                    "Set `download=True` or provide a valid local cache."
                )
            dataset_path = "bigbio/pubmed_qa"
            is_local = False

            if self._root:
                logger.info(f"Dataset will be downloaded and cached in: {self._root}")
            else:
                logger.info("Using dataset directly from HuggingFace without caching.")

        split = (self._split or "train+test+validation") if self._split != "val" else "validation"
        raw_dataset = load_dataset(
            dataset_path,
            name="pubmed_qa_labeled_fold0_source",
            split=split,
            streaming=False,
            cache_dir=self._root if (not is_local and self._root) else None,
            trust_remote_code=True,
        )
        if not isinstance(raw_dataset, Dataset):
            raise TypeError(f"Expected a `Dataset`, but got {type(raw_dataset)}")

        return raw_dataset

    @override
    def prepare_data(self) -> None:
        """Downloads and prepares the PubMedQA dataset.

        If `self._root` is None, the dataset is used directly from HuggingFace.
        Otherwise, it checks if the dataset is already cached in `self._root`.
        If not cached, it downloads the dataset into `self._root`.
        """
        dataset_cache_path = None

        if self._root:
            dataset_cache_path = os.path.join(self._root, "pubmed_qa")
            os.makedirs(self._root, exist_ok=True)

        try:
            self.dataset = self._load_dataset(dataset_cache_path)
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
        sample = dict(self.dataset[index])
        return f"Question: {sample['QUESTION']} \nContext: {sample['CONTEXTS']}"

    @override
    def load_target(self, index: int) -> torch.Tensor:
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
