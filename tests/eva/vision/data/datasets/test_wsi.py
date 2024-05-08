"""WsiDataset & MultiWsiDataset tests."""

import os
from typing import Tuple

import pytest

from eva.vision.data import datasets
from eva.vision.data.wsi.patching import samplers


@pytest.mark.parametrize(
    "width, height, overlap",
    [
        (4, 4, (0, 0)),
        (4, 4, (2, 2)),
        (33, 33, (0, 0)),
        (224, 224, (0, 0)),
    ],
)
def test_len(width: int, height: int, root: str, overlap: Tuple[int, int]):
    """Test the length of the dataset using different patch dimensions."""
    dataset = datasets.WsiDataset(
        file_path=os.path.join(root, "0/a.tiff"),
        width=width,
        height=height,
        target_mpp=0.25,
        sampler=samplers.GridSampler(max_samples=None, overlap=overlap),
        backend="openslide",
    )

    layer_shape = dataset._wsi.level_dimensions[0]
    assert len(dataset) == _expected_n_patches(layer_shape, width, height, overlap)


@pytest.mark.parametrize(
    "width, height, target_mpp",
    [(4, 4, 0.25), (4, 4, 1.3)],
)
def test_patch_shape(width: int, height: int, target_mpp: float, root: str):
    """Test the shape of the extracted patches."""
    dataset = datasets.WsiDataset(
        file_path=os.path.join(root, "0/a.tiff"),
        width=width,
        height=height,
        target_mpp=target_mpp,
        sampler=samplers.GridSampler(max_samples=None),
        backend="openslide",
    )

    mpp_ratio = target_mpp / (
        dataset._wsi.mpp * dataset._wsi.level_downsamples[dataset._coords.level_idx]
    )
    scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)
    assert dataset[0].shape == (scaled_width, scaled_height, 3)


def test_multi_dataset(root: str):
    """Test MultiWsiDataset with multiple whole-slide image paths."""
    file_paths = [
        os.path.join(root, "0/a.tiff"),
        os.path.join(root, "0/b.tiff"),
        os.path.join(root, "1/a.tiff"),
    ]

    width, height = 32, 32
    dataset = datasets.MultiWsiDataset(
        root=root,
        file_paths=file_paths,
        width=width,
        height=height,
        target_mpp=0.25,
        sampler=samplers.GridSampler(max_samples=None),
        backend="openslide",
    )

    assert isinstance(dataset.datasets[0], datasets.WsiDataset)
    layer_shape = dataset.datasets[0]._wsi.level_dimensions[0]
    assert len(dataset) == _expected_n_patches(layer_shape, width, height, (0, 0)) * len(file_paths)
    assert dataset.cumulative_sizes == [64, 128, 192]


def _expected_n_patches(layer_shape, width, height, overlap):
    """Calculate the expected number of patches."""
    n_patches_x = (layer_shape[0] - width) // (width - overlap[0]) + 1
    n_patches_y = (layer_shape[1] - height) // (height - overlap[1]) + 1
    return n_patches_x * n_patches_y


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root path to the test dataset assets."""
    return os.path.join(assets_path, "vision/datasets/wsi")
