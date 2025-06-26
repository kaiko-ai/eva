"""PANDA dataset tests."""

import os
from typing import Literal
from unittest.mock import PropertyMock, patch

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets
from eva.vision.data.wsi.patching import samplers


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 4), ("val", 3), (None, 7)],
)
def test_length(dataset: datasets.CoNSeP, expected_length: int) -> None:
    """Tests the length of the dataset."""
    # 16 patches (10x10) per slide (40x40)
    assert len(dataset) == expected_length * 16


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
        ("train", 0),
        ("val", 0),
    ],
)
def test_sample(dataset: datasets.CoNSeP, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, 10, 10)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (10, 10)
    assert isinstance(metadata, dict)
    assert "coords" in metadata


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root directory of the dataset."""
    return os.path.join(assets_path, "vision/datasets/consep")


@pytest.fixture(scope="function")
def dataset(split: Literal["train", "val"] | None, root: str) -> datasets.CoNSeP:
    """CoNSeP dataset fixture."""
    with patch.object(
        datasets.CoNSeP,
        "_expected_dataset_lengths",
        new_callable=PropertyMock(return_value={None: 7, "train": 4, "val": 3}),
    ):
        dataset = datasets.CoNSeP(
            root=root,
            split=split,
            width=10,
            height=10,
            target_mpp=0.25,
            sampler=samplers.GridSampler(),
        )
        dataset.prepare_data()
        dataset.configure()
        return dataset
