"""Tests for the balanced sampler."""

from collections import Counter

import pytest
import torch

from eva.core.data.datasets.typings import DataSample
from eva.core.data.samplers.classification import BalancedSampler
from tests.eva.core.data.samplers import _utils


@pytest.mark.parametrize(
    "num_class_samples, replacement, num_dataset_samples, num_classes",
    [
        (3, False, 15, 2),
        (20, True, 15, 2),
        (3, False, 33, 5),
    ],
)
def test_balanced_sampling(
    num_class_samples: int, replacement: bool, num_dataset_samples: int, num_classes: int
):
    """Tests if the returned indices are balanced."""
    dataset = _utils.multiclass_dataset(num_dataset_samples, num_classes)
    sampler = BalancedSampler(num_samples=num_class_samples, replacement=replacement)
    sampler.set_dataset(dataset)

    indices = list(sampler)
    class_counts = Counter(DataSample(*dataset[i]).targets.item() for i in indices)  # type: ignore

    assert len(sampler) == num_class_samples * num_classes
    assert len(class_counts.keys()) == num_classes
    for count in class_counts.values():
        assert count == num_class_samples


def test_insufficient_samples_without_replacement():
    """Tests if the sampler raises an error when there are insufficient samples."""
    num_dataset_samples, num_classes = 15, 3
    dataset = _utils.multiclass_dataset(num_dataset_samples, num_classes)
    sampler = BalancedSampler(num_samples=7, replacement=False)

    with pytest.raises(ValueError, match=f"has only {num_dataset_samples // num_classes} samples"):
        sampler.set_dataset(dataset)


def test_random_seed():
    """Tests if the sampler is reproducible with the same seed."""
    num_dataset_samples, num_classes = 101, 3
    dataset = _utils.multiclass_dataset(num_dataset_samples, num_classes)
    sampler1 = BalancedSampler(num_samples=10, seed=1)
    sampler1_duplicate = BalancedSampler(num_samples=10, seed=1)
    sampler2 = BalancedSampler(num_samples=10, seed=2)
    sampler1.set_dataset(dataset)
    sampler1_duplicate.set_dataset(dataset)
    sampler2.set_dataset(dataset)

    assert list(sampler1) == list(sampler1_duplicate)
    assert list(sampler1) != list(sampler2)


def test_invalid_targets():
    """Tests if the sampler raises an error unsupported target formats."""
    sampler = BalancedSampler(num_samples=10)

    # test multi-dimensional target
    dataset = _utils.MockDataset([(None, torch.tensor([0, 1]), None)])
    with pytest.raises(ValueError, match="single & scalar target"):
        sampler.set_dataset(dataset)

    # test empty target
    dataset = _utils.MockDataset([(None, None, None)])  # type: ignore
    with pytest.raises(ValueError, match="non-empty targets"):
        sampler.set_dataset(dataset)
