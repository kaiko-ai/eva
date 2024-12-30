"""PubMedQA dataset class."""

from typing import Any, Dict, List
import torch
from datasets import load_dataset, Dataset
from typing_extensions import override

from eva.language.data.datasets.classification import base


class PubMedQA(base.TextClassification):
    """Dataset class for PubMedQA question answering task."""

    _license: str = "MIT License (https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE)"
    """Dataset license."""

    def __init__(
        self,
        split: str | None = "train+test+validation",
    ) -> None:
        """Initialize the PubMedQA dataset.

        Args:
            split: Dataset split to use. If default, entire dataset of 1000 samples is used.
        """
        super().__init__()
        self._split = split
        raw_dataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold0_source",
            split=split,
            streaming=False
        )

        if not isinstance(raw_dataset, Dataset):
            raise TypeError(f"Expected a `Dataset`, but got {type(raw_dataset)}")

        self.dataset: Dataset = raw_dataset

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
            self.class_to_idx[self.dataset[index]["final_decision"]],
            dtype=torch.long
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
