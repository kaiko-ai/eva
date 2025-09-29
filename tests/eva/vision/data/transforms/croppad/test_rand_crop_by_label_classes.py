"""Tests for RandCropByLabelClasses transform."""

import pytest
import torch
from torchvision import tv_tensors

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.transforms.croppad.rand_crop_by_label_classes import RandCropByLabelClasses

SPATIAL_SIZE = (8, 16, 16)
CROP_SIZE = (4, 4, 4)
NUM_CLASSES = 3
NUM_SAMPLES = 5


@pytest.fixture
def volume() -> eva_tv_tensors.Volume:
    """Creates a 3D volume tensor of shape (1, D, H, W)."""
    return eva_tv_tensors.Volume(torch.randn((1,) + SPATIAL_SIZE, dtype=torch.float32))


@pytest.fixture
def mask(num_classes: int = NUM_CLASSES) -> tv_tensors.Mask:
    """Creates a 3D mask tensor of shape (1, D, H, W) with random class labels."""
    return tv_tensors.Mask(torch.randint(0, num_classes, (1,) + SPATIAL_SIZE, dtype=torch.int64))


def test_rand_crop_returns_expected_num_samples(
    volume: eva_tv_tensors.Volume, mask: tv_tensors.Mask
) -> None:
    """Transform returns the requested number of crops and wraps them in tv_tensors."""
    transform = RandCropByLabelClasses(
        spatial_size=CROP_SIZE,
        ratios=[1, 1, 1],
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
    )
    volume_crops, mask_crops = transform(volume, mask)

    assert len(volume_crops) == len(mask_crops) == NUM_SAMPLES
    assert all(isinstance(crop, eva_tv_tensors.Volume) for crop in volume_crops)
    assert all(isinstance(crop, tv_tensors.Mask) for crop in mask_crops)
    assert all(crop.shape[-3:] == CROP_SIZE for crop in volume_crops)
    assert all(crop.shape[-3:] == CROP_SIZE for crop in mask_crops)


def test_rand_crop_reproducible_with_same_seed(
    volume: eva_tv_tensors.Volume, mask: tv_tensors.Mask
) -> None:
    """Transforms with the same seed should produce identical crops."""
    transform_a = RandCropByLabelClasses(
        spatial_size=CROP_SIZE,
        ratios=[1, 1, 1],
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
    )
    transform_b = RandCropByLabelClasses(
        spatial_size=CROP_SIZE,
        ratios=[1, 1, 1],
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
    )

    transform_a.set_random_state(5)
    transform_b.set_random_state(5)

    volume_crops_a, mask_crops_a = transform_a(volume, mask)
    volume_crops_b, mask_crops_b = transform_b(volume, mask)

    assert len(volume_crops_a) == len(volume_crops_b)
    assert len(mask_crops_a) == len(mask_crops_b)

    for crop_a, crop_b in zip(volume_crops_a, volume_crops_b, strict=False):
        assert torch.equal(crop_a, crop_b)

    for crop_a, crop_b in zip(mask_crops_a, mask_crops_b, strict=False):
        assert torch.equal(crop_a, crop_b)
