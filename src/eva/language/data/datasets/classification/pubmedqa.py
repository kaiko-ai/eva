"""PubMedQA dataset class."""

import os
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from typing_extensions import override

from eva.language.data.datasets.classification import base


class PubMedQA(base.TextClassification):
    """Dataset class for PubMedQA question answering task."""

    _license: str = "MIT License (https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE)"
    """Dataset license."""

    def __init__(
        self,
        root: str | None = None,
        split: str | None = "train+test+validation",
        download: bool = False,
    ) -> None:
        """Initialize the PubMedQA dataset.

        Args:
            root: Directory to cache the dataset. If None, no local caching is used.
            split: Dataset split to use. Default is "train+test+validation".
            download: Whether to download the dataset if not found locally. Default is False.
        """
        super().__init__()
        self._root = root
        self._split = split
        self._download = download

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
            if dataset_cache_path and os.path.exists(dataset_cache_path):
                raw_dataset = load_dataset(
                    dataset_cache_path,
                    name="pubmed_qa_labeled_fold0_source",
                    split=self._split,
                    streaming=False,
                )
                print(f"Loaded dataset from local cache: {dataset_cache_path}")
            else:
                if not self._download and self._root:
                    raise ValueError(
                        "Dataset not found locally and downloading is disabled. "
                        "Set `download=True` or provide a valid local cache."
                    )

                raw_dataset = load_dataset(
                    "bigbio/pubmed_qa",
                    name="pubmed_qa_labeled_fold0_source",
                    split=self._split,
                    cache_dir=self._root if self._root else None,
                    streaming=False,
                )
                if self._root:
                    print(f"Dataset downloaded and cached in: {self._root}")
                else:
                    print("Using dataset directly from Hugging Face without caching.")

            if not isinstance(raw_dataset, Dataset):
                raise TypeError(f"Expected a `Dataset`, but got {type(raw_dataset)}")

            self.dataset: Dataset = raw_dataset

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
        return f"Question: {sample['QUESTION']}\nContext: {sample['CONTEXTS']}"

    @override
    def load_target(self, index: int) -> torch.Tensor:
        return torch.tensor(
            self.class_to_idx[self.dataset[index]["final_decision"]], dtype=torch.long
        )

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        return {
            "year": sample["YEAR"],
            "labels": sample["LABELS"],
            "meshes": sample["MESHES"],
            "long_answer": sample["LONG_ANSWER"],
            "reasoning_required": sample["reasoning_required_pred"],
            "reasoning_free": sample["reasoning_free_pred"],
        }

    @override
    def __len__(self) -> int:
        return len(self.dataset)

    def _print_license(self) -> None:
        """Prints the dataset license."""
        print(f"Dataset license: {self._license}")
