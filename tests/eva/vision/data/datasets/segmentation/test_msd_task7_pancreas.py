"""MSDTask7Pancreas dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets
from eva.vision.data import tv_tensors as eva_tv_tensors


@pytest.mark.parametrize(
    "split, expected_length",
    [(None, 2), ("train", 1), ("val", 1)],
)
def test_length(
    msd_task7_pancreas_dataset: datasets.MSDTask7Pancreas, expected_length: int
) -> None:
    """Tests the length of the dataset."""
    assert len(msd_task7_pancreas_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(msd_task7_pancreas_dataset: datasets.MSDTask7Pancreas, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = msd_task7_pancreas_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, eva_tv_tensors.Volume)
    assert image.shape == (4, 1, 8, 8)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (4, 1, 8, 8)
    assert isinstance(metadata, dict)


@pytest.fixture(scope="function")
def msd_task7_pancreas_dataset(
    split: Literal["train", "val"] | None, assets_path: str
) -> datasets.MSDTask7Pancreas:
    """MSDTask7Pancreas dataset fixture."""
    dataset = datasets.MSDTask7Pancreas(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "msd_task7_pancreas",
        ),
        split=split,
    )

    dataset._train_ids = [1]
    dataset._val_ids = [6]

    dataset.prepare_data()
    dataset.configure()
    return dataset
