"""Fake dataset that returns dictionaries."""

from typing import Tuple

import torch
from typing_extensions import override

from eva.data import datasets
from eva.models.modules.typings import DICT_INPUT_BATCH


class FakeDictDataset(datasets.Dataset[DICT_INPUT_BATCH]):
    """Dataset that returns a dictionary with random data."""

    def __init__(
        self,
        n_samples: int,
        data_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...] = (),
        n_classes: int = 4,
    ):
        """Initializes the dataset."""
        self.data = torch.randn(n_samples, *data_shape)
        self.targets = torch.randint(n_classes, (n_samples,) + target_shape, dtype=torch.long)
        self.metadata = [{f"field_{i}": i for i in range(2)}] * n_samples

    def __len__(self):
        """Returns the dataset length."""
        return len(self.data)

    @override
    def __getitem__(self, index) -> DICT_INPUT_BATCH:
        sample = {
            "data": self.data[index],
            "targets": self.targets[index],
            "metadata": self.metadata[index],
        }
        return sample  # type: ignore
