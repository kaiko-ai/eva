"""Test utilities for dataloader sampler tests."""

from typing import List, Tuple

import torch
from typing_extensions import override

from eva.core.data import datasets


class MockDataset(datasets.MapDataset):
    """Mock map-style dataset class for unit testing."""

    def __init__(self, samples: List[Tuple[None, torch.Tensor, None]]):
        self.samples = samples

    @override
    def __getitem__(self, idx):
        return self.samples[idx]

    @override
    def __len__(self):
        return len(self.samples)


def multiclass_dataset(num_samples: int, num_classes: int) -> datasets.MapDataset:
    samples = (
        [(None, torch.tensor([i]), None)] * (num_samples // num_classes) for i in range(num_classes)
    )
    return MockDataset([item for sublist in samples for item in sublist])
