"""Base for text classification datasets."""

import abc
from typing import Any, Dict, List, Tuple

import torch
from typing_extensions import override

from eva.language.data.datasets.language import LanguageDataset


class TextClassification(LanguageDataset[Tuple[str, torch.Tensor, Dict[str, Any]]], abc.ABC):
    """Text classification abstract dataset."""

    def __init__(self) -> None:
        """Initializes the text classification dataset."""
        super().__init__()

    @property
    def classes(self) -> List[str] | None:
        """Returns list of class names."""

    @property
    def class_to_idx(self) -> Dict[str, int] | None:
        """Returns class name to index mapping."""

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the dataset metadata.

        Args:
            index: The index of the data sample.

        Returns:
            The sample metadata.
        """

    @abc.abstractmethod
    def load_text(self, index: int) -> str:
        """Returns the text content.

        Args:
            index: The index of the data sample.

        Returns:
            The text content.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_target(self, index: int) -> torch.Tensor:
        """Returns the target label.

        Args:
            index: The index of the data sample.

        Returns:
            The target label.
        """
        raise NotImplementedError

    @override
    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, Dict[str, Any]]:
        return (self.load_text(index), self.load_target(index), self.load_metadata(index) or {})
