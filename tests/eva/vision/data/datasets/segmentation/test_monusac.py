"""MoNuSac dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 5), ("test", 4)],
)
def test_length(monusac_dataset: datasets.MoNuSAC, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(monusac_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index, expected_height,  expected_width",
    [
        ("train", 0, 512, 512),
        ("test", 0, 142, 115),
    ],
)
def test_sample(
    monusac_dataset: datasets.MoNuSAC, index: int, expected_height: int, expected_width: int
) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = monusac_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (3, expected_height, expected_width)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (expected_height, expected_width)
    assert isinstance(metadata, dict)


@pytest.fixture(scope="function")
def monusac_dataset(split: Literal["train", "test"], assets_path: str) -> datasets.MoNuSAC:
    """MoNuSAC dataset fixture."""
    dataset = datasets.MoNuSAC(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "monusac",
        ),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
