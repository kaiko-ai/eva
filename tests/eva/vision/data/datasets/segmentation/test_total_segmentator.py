"""TotalSegmentator2D dataset tests."""

import os
import shutil
from typing import Dict, Literal
from unittest.mock import patch

import pytest
from torchvision import tv_tensors

from eva.vision.data import datasets

_class_mappings = {
    "aorta_small": "class_1",
    "brain_small": "class_1",
    "colon_small": "class_2",
}


@pytest.mark.parametrize(
    "split, expected_length, class_mappings, optimize_mask_loading",
    [
        ("train", 6, None, False),
        ("train", 6, None, True),
        ("val", 3, None, False),
        (None, 9, None, False),
    ],
)
def test_length(
    total_segmentator_dataset: datasets.TotalSegmentator2D, expected_length: int
) -> None:
    """Tests the length of the dataset."""
    assert len(total_segmentator_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index, class_mappings, optimize_mask_loading",
    [
        (None, 0, None, False),
        (None, 0, None, True),
        ("train", 0, None, False),
        ("val", 0, None, False),
        ("train", 0, _class_mappings, False),
    ],
)
def test_sample(total_segmentator_dataset: datasets.TotalSegmentator2D, index: int) -> None:
    """Tests the format of a dataset sample."""
    # assert data sample is a tuple
    sample = total_segmentator_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    # assert the format of the `image` and `mask`
    image, mask, metadata = sample
    assert isinstance(image, tv_tensors.Image)
    assert image.shape == (1, 16, 16)
    assert isinstance(mask, tv_tensors.Mask)
    assert mask.shape == (16, 16)
    assert isinstance(metadata, dict)
    assert "slice_index" in metadata

    # check the number of classes with v.s. without class mappings
    n_classes_expected = 3 if total_segmentator_dataset._class_mappings is not None else 4
    assert len(total_segmentator_dataset.classes) == n_classes_expected


@pytest.mark.parametrize(
    "split, class_mappings, optimize_mask_loading",
    [
        ("train", None, False),
        ("train", None, True),
        ("train", _class_mappings, True),
    ],
)
def test_optimize_mask_loading(total_segmentator_dataset: datasets.TotalSegmentator2D):
    """Tests the optimized mask loading."""
    optimize = total_segmentator_dataset._optimize_mask_loading is True

    if optimize:
        expected_masks_dir = os.path.join(
            total_segmentator_dataset._root,
            f"processed/masks/{total_segmentator_dataset._classes_hash}",
        )
        expected_classes_file = os.path.join(expected_masks_dir, "classes.txt")
        assert os.path.isdir(expected_masks_dir)
        assert os.path.isfile(expected_classes_file)

        with open(expected_classes_file, "r") as f:
            assert f.read() == str(total_segmentator_dataset.classes)

    with (
        patch.object(total_segmentator_dataset, "_load_semantic_label_mask") as mock_load_optimized,
        patch.object(total_segmentator_dataset, "_load_target") as mock_load,
        patch.object(total_segmentator_dataset, "_fix_orientation") as _,
    ):
        _ = total_segmentator_dataset.load_target(0)
        if optimize:
            mock_load_optimized.assert_called_once_with(0)
            mock_load.assert_not_called()
        else:
            mock_load.assert_called_once_with(0)
            mock_load_optimized.assert_not_called()


@pytest.fixture(scope="function")
def total_segmentator_dataset(
    tmp_path: str,
    split: Literal["train", "val"] | None,
    assets_path: str,
    class_mappings: Dict[str, str] | None,
    optimize_mask_loading: bool,
) -> datasets.TotalSegmentator2D:
    """TotalSegmentator2D dataset fixture."""
    dataset_dir = os.path.join(
        assets_path,
        "vision",
        "datasets",
        "total_segmentator",
        "Totalsegmentator_dataset_v201",
    )
    shutil.copytree(dataset_dir, tmp_path, dirs_exist_ok=True)
    dataset = datasets.TotalSegmentator2D(
        root=tmp_path,
        split=split,
        version=None,
        class_mappings=class_mappings,
        optimize_mask_loading=optimize_mask_loading,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset
