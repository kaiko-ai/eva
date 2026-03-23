"""BCSS dataset tests."""

import os
from typing import Literal
from unittest.mock import PropertyMock, patch

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets
from eva.vision.data.wsi.patching import samplers


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 72), ("val", 12), ("trainval", 84), ("test", 40), (None, 124)],
)
def test_length(dataset: datasets.BCSS, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("val", 0),
        ("trainval", 0),
        ("test", 0),
        (None, 0),
    ],
)
def test_sample(dataset: datasets.BCSS, index: int) -> None:
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
    assert (0 <= mask).all() and (mask <= 5).all()
    assert isinstance(metadata, dict)
    assert "coords" in metadata


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root directory of the dataset."""
    return os.path.join(assets_path, "vision/datasets/bcss")


@pytest.fixture(scope="function")
def dataset(split: Literal["train", "val"] | None, root: str) -> datasets.BCSS:
    """BCSS dataset fixture."""
    with patch.object(
        datasets.BCSS,
        "_expected_length",
        new_callable=PropertyMock(return_value=10),
    ):
        dataset = datasets.BCSS(
            root=root,
            split=split,
            width=10,
            height=10,
            target_mpp=0.25,
            sampler=samplers.GridSampler(),
        )
        dataset.prepare_data()
        dataset.setup()
        return dataset
