"""WsiDataset & MultiWsiDataset tests."""

import os
import pathlib
from typing import Tuple

import pandas as pd
import pytest

from eva.vision.data import datasets
from eva.vision.data.wsi.backends import is_backend_available
from eva.vision.data.wsi.patching import samplers


@pytest.mark.parametrize(
    "width, height, overlap, backend",
    [
        (4, 4, (0, 0), "openslide"),
        (4, 4, (2, 2), "openslide"),
        (33, 33, (0, 0), "openslide"),
        (224, 224, (0, 0), "openslide"),
        (4, 4, (0, 0), "tiffslide"),
        (4, 4, (2, 2), "tiffslide"),
        (33, 33, (0, 0), "tiffslide"),
        (224, 224, (0, 0), "tiffslide"),
    ],
)
def test_len(width: int, height: int, root: str, overlap: Tuple[int, int], backend: str):
    """Test the length of the dataset using different patch dimensions."""
    if not is_backend_available(backend):
        pytest.skip(f"{backend} backend is not available.")
    dataset = datasets.WsiDataset(
        file_path=os.path.join(root, "0/a.tiff"),
        width=width,
        height=height,
        target_mpp=0.25,
        sampler=samplers.GridSampler(max_samples=None, overlap=overlap),
        backend=backend,
    )

    layer_shape = dataset._wsi.level_dimensions[0]
    assert len(dataset) == _expected_n_patches(layer_shape, width, height, overlap)


@pytest.mark.parametrize(
    "width, height, target_mpp, backend",
    [
        (4, 4, 0.25, "openslide"),
        (4, 4, 1.3, "openslide"),
        (4, 4, 0.25, "tiffslide"),
        (4, 4, 1.3, "tiffslide"),
    ],
)
def test_patch_shape(width: int, height: int, target_mpp: float, root: str, backend: str):
    """Test the shape of the extracted patches."""
    if not is_backend_available(backend):
        pytest.skip(f"{backend} backend is not available.")
    dataset = datasets.WsiDataset(
        file_path=os.path.join(root, "0/a.tiff"),
        width=width,
        height=height,
        target_mpp=target_mpp,
        sampler=samplers.GridSampler(max_samples=None),
        backend=backend,
    )

    mpp_ratio = target_mpp / (
        dataset._wsi.mpp * dataset._wsi.level_downsamples[dataset._coords.level_idx]
    )
    scaled_width, scaled_height = int(mpp_ratio * width), int(mpp_ratio * height)
    assert dataset[0].shape == (3, scaled_width, scaled_height)


def test_multi_dataset(root: str, tmp_path: pathlib.Path):
    """Test MultiWsiDataset with multiple whole-slide image paths."""
    coords_path = (tmp_path / "coords.csv").as_posix()
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
        coords_path=coords_path,
    )
    dataset.setup()

    assert isinstance(dataset.datasets[0], datasets.WsiDataset)
    layer_shape = dataset.datasets[0]._wsi.level_dimensions[0]
    assert len(dataset) == _expected_n_patches(layer_shape, width, height, (0, 0)) * len(file_paths)
    assert dataset.cumulative_sizes == [64, 128, 192]

    assert os.path.exists(coords_path)
    df_coords = pd.read_csv(coords_path)
    assert "file" in df_coords.columns
    assert "x_y" in df_coords.columns


def _expected_n_patches(layer_shape, width, height, overlap):
    """Calculate the expected number of patches."""
    n_patches_x = (layer_shape[0] - width) // (width - overlap[0]) + 1
    n_patches_y = (layer_shape[1] - height) // (height - overlap[1]) + 1
    return n_patches_x * n_patches_y


@pytest.fixture
def root(assets_path: str) -> str:
    """Fixture returning the root path to the test dataset assets."""
    return os.path.join(assets_path, "vision/datasets/wsi")
