"""KiTS23 dataset tests."""

import os
import shutil
from typing import Literal
from unittest.mock import patch

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [(None, 8)],
)
def test_length(kits23_dataset: datasets.KiTS23, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(kits23_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        (None, 0),
    ],
)
def test_sample(kits23_dataset: datasets.KiTS23, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = kits23_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (1, 512, 512)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (512, 512)
    assert isinstance(metadata, dict)
    assert "slice_index" in metadata


@pytest.mark.parametrize("split", [None])
def test_processed_dir_exists(kits23_dataset: datasets.KiTS23) -> None:
    """Tests the existence of the processed directory."""
    assert os.path.isdir(kits23_dataset._processed_root)

    for index in ["00036", "00240"]:
        assert os.path.isfile(
            os.path.join(kits23_dataset._processed_root, f"case_{index}/master_{index}.nii")
        )
        assert os.path.isfile(
            os.path.join(kits23_dataset._processed_root, f"case_{index}/segmentation.nii")
        )


@pytest.fixture(scope="function")
def kits23_dataset(split: Literal["train", "val", "test"] | None, assets_path: str):
    """KiTS23 dataset fixture."""
    dataset = datasets.KiTS23(
        root=os.path.join(
            assets_path,
            "vision",
            "datasets",
            "kits23",
        ),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    yield dataset

    if os.path.isdir(dataset._processed_root):
        shutil.rmtree(dataset._processed_root)


@pytest.fixture(autouse=True)
def mock_indices():
    """Mocks the download function to avoid downloading resources when running tests."""
    with patch.object(datasets.KiTS23, "_get_split_indices", return_value=[36, 240]):
        yield
