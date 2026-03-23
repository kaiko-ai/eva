"""Tests for the stratified random sampler."""

from collections import Counter

import torch

from eva.core.data.datasets.typings import DataSample
from eva.core.data.samplers.classification import StratifiedRandomSampler
from tests.eva.core.data.samplers import _utils


def test_stratified_sampling():
    """Test that class proportions are maintained in stratified sampling."""
    dataset = _utils.multiclass_dataset(num_samples=60, num_classes=3)  # 20 samples per class
    sampler = StratifiedRandomSampler(num_samples=30, seed=42)  # Sample 50%
    sampler.set_dataset(dataset)

    indices = list(sampler)
    class_counts = Counter(DataSample(*dataset[i]).targets.item() for i in indices)  # type: ignore

    assert len(indices) == 30
    for class_id in range(3):
        assert 9 <= class_counts[class_id] <= 11  # ~10 samples per class


def test_stratified_sampling_with_exact_proportions():
    """Test stratified sampling with a perfectly balanced dataset."""
    dataset = _utils.multiclass_dataset(num_samples=100, num_classes=4)  # 25 samples per class
    sampler = StratifiedRandomSampler(num_samples=40, seed=42)  # Sample 40%
    sampler.set_dataset(dataset)

    indices = list(sampler)
    class_counts = Counter(DataSample(*dataset[i]).targets.item() for i in indices)  # type: ignore

    assert len(indices) == 40
    for class_id in range(4):
        assert class_counts[class_id] == 10  # Exactly 10 samples per class


def test_stratified_sampling_with_imbalanced_dataset():
    """Test stratified sampling with an imbalanced dataset."""
    samples_0 = [(None, torch.tensor([0]), None)] * 50
    samples_1 = [(None, torch.tensor([1]), None)] * 30
    samples_2 = [(None, torch.tensor([2]), None)] * 20
    dataset = _utils.MockDataset(samples_0 + samples_1 + samples_2)

    sampler = StratifiedRandomSampler(num_samples=50, seed=42)  # Sample 50%
    sampler.set_dataset(dataset)

    indices = list(sampler)
    class_counts = Counter(DataSample(*dataset[i]).targets.item() for i in indices)  # type: ignore

    assert len(indices) == 50
    assert 24 <= class_counts[0] <= 26  # ~25 samples for class 0
    assert 14 <= class_counts[1] <= 16  # ~15 samples for class 1
    assert 9 <= class_counts[2] <= 11  # ~10 samples for class 2


def test_sample_ratio():
    """Test sampling with sample_ratio parameter."""
    dataset = _utils.multiclass_dataset(num_samples=120, num_classes=3)
    sampler = StratifiedRandomSampler(sample_ratio=0.5, seed=42)
    sampler.set_dataset(dataset)

    indices = list(sampler)
    class_counts = Counter(DataSample(*dataset[i]).targets.item() for i in indices)  # type: ignore

    assert len(indices) == 60
    for class_id in range(3):
        assert 19 <= class_counts[class_id] <= 21  # ~20 samples per class


def test_sample_ratio_with_rounding():
    """Test that sample_ratio rounds correctly."""
    dataset = _utils.multiclass_dataset(num_samples=27, num_classes=3)
    sampler = StratifiedRandomSampler(sample_ratio=0.5, seed=42)
    sampler.set_dataset(dataset)

    indices = list(sampler)
    assert len(indices) == 14  # 27 * 0.5 = 13.5


def test_num_samples_takes_precedence_over_ratio():
    """Test that num_samples takes precedence when both are provided."""
    dataset = _utils.multiclass_dataset(num_samples=120, num_classes=3)
    sampler = StratifiedRandomSampler(num_samples=30, sample_ratio=0.5, seed=42)
    sampler.set_dataset(dataset)

    indices = list(sampler)
    assert len(indices) == 30


def test_no_params_uses_full_dataset():
    """Test that not providing num_samples or sample_ratio uses the full dataset."""
    dataset = _utils.multiclass_dataset(num_samples=60, num_classes=3)
    sampler = StratifiedRandomSampler(seed=42)
    sampler.set_dataset(dataset)

    indices = list(sampler)
    assert len(indices) == 60


def test_random_seed():
    """Test reproducibility with same seed."""
    dataset = _utils.multiclass_dataset(num_samples=60, num_classes=3)
    sampler1 = StratifiedRandomSampler(num_samples=20, seed=42)
    sampler1_duplicate = StratifiedRandomSampler(num_samples=20, seed=42)
    sampler2 = StratifiedRandomSampler(num_samples=20, seed=123)

    sampler1.set_dataset(dataset)
    sampler1_duplicate.set_dataset(dataset)
    sampler2.set_dataset(dataset)

    assert list(sampler1) == list(sampler1_duplicate)
    assert list(sampler1) != list(sampler2)
