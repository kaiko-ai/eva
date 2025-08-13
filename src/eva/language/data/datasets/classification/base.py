"""Base for text classification datasets."""

from typing import Dict, List

import torch

from eva.language.data.datasets.text import TextDataset


class TextClassification(TextDataset[torch.Tensor]):
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
