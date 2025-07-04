"""PubMedQA dataset tests."""

import os
import shutil

import pytest
import torch
from datasets import Dataset

from eva.language.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 450), ("test", 500), ("val", 50), (None, 1000)],
)
def test_length(pubmedqa_dataset: datasets.PubMedQA, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(pubmedqa_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("train", 10),
        ("test", 0),
        ("val", 0),
        (None, 0),
    ],
)
def test_sample(pubmedqa_dataset: datasets.PubMedQA, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = pubmedqa_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3

    text, target, metadata = sample
    assert isinstance(text, str)
    assert text.startswith("Question: ")
    assert "Context: " in text

    assert isinstance(target, torch.Tensor)
    assert target in [0, 1, 2]

    assert isinstance(metadata, dict)
    required_keys = {
        "year",
        "labels",
        "meshes",
        "long_answer",
        "reasoning_required",
        "reasoning_free",
    }
    assert all(key in metadata for key in required_keys)


@pytest.mark.parametrize("split", [None])
def test_classes(pubmedqa_dataset: datasets.PubMedQA) -> None:
    """Tests the dataset classes."""
    assert pubmedqa_dataset.classes == ["no", "yes", "maybe"]
    assert pubmedqa_dataset.class_to_idx == {"no": 0, "yes": 1, "maybe": 2}


@pytest.mark.parametrize("split", [None])
def test_prepare_data_no_root(pubmedqa_dataset: datasets.PubMedQA) -> None:
    """Tests dataset preparation without specifying a root directory."""
    assert isinstance(pubmedqa_dataset.dataset, Dataset)
    assert len(pubmedqa_dataset) > 0


@pytest.mark.parametrize("split", [None])
def test_prepare_data_with_cache(pubmedqa_dataset_with_cache: datasets.PubMedQA) -> None:
    """Tests dataset preparation with caching."""
    pubmedqa_dataset_with_cache.prepare_data()
    assert isinstance(pubmedqa_dataset_with_cache.dataset, Dataset)
    assert len(pubmedqa_dataset_with_cache) > 0

    cache_dir = pubmedqa_dataset_with_cache._root
    if cache_dir:
        assert os.path.exists(cache_dir)
        assert any(os.scandir(cache_dir))


@pytest.mark.parametrize("split", [None])
def test_prepare_data_without_download(tmp_path, split) -> None:
    """Tests dataset preparation when download is disabled and cache is missing."""
    dataset = datasets.PubMedQA(split=split, download=False)

    with pytest.raises(RuntimeError, match="Failed to prepare dataset: Dataset path not found."):
        dataset.prepare_data()


def test_cleanup_cache(tmp_path) -> None:
    """Tests that the cache can be cleaned up."""
    root = tmp_path / "pubmed_qa_cache"
    dataset = datasets.PubMedQA(root=str(root), download=True)
    dataset.prepare_data()

    assert os.path.exists(root)

    shutil.rmtree(root)
    assert not os.path.exists(root)


@pytest.fixture(scope="function")
def pubmedqa_dataset(split: None) -> datasets.PubMedQA:
    """PubMedQA dataset fixture."""
    dataset = datasets.PubMedQA(split=split, download=True)
    dataset.prepare_data()
    return dataset


@pytest.fixture(scope="function")
def pubmedqa_dataset_with_cache(tmp_path, split: None) -> datasets.PubMedQA:
    """PubMedQA dataset fixture with caching enabled."""
    root = tmp_path / "pubmed_qa_cache"
    dataset = datasets.PubMedQA(root=str(root), split=split, download=True)
    dataset.prepare_data()
    return dataset
