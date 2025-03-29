"""Tests for the nifti IO functions."""

import os

import nibabel as nib
import numpy as np
import pytest
from nibabel import orientations

from eva.vision.utils.io import nifti


@pytest.mark.parametrize(
    "use_storage_dtype, target_orientation",
    [
        [False, None],
        [False, "LPS"],
        [True, "RAS"],
    ],
)
def test_read_nifti(nifti_path: str, use_storage_dtype: bool, target_orientation: str):
    """Tests the function to read a nifti file as array (full & slice)."""
    image = nifti.read_nifti(
        nifti_path, use_storage_dtype=use_storage_dtype, target_orientation=target_orientation
    )
    assert image.shape == (512, 512, 4)

    slice_image = nifti.read_nifti(
        nifti_path,
        slice_index=0,
        use_storage_dtype=use_storage_dtype,
        target_orientation=target_orientation,
    )
    assert slice_image.shape == (512, 512, 1)

    expected_dtype = np.dtype("<i2") if use_storage_dtype else np.float_
    assert image.dtype == expected_dtype


@pytest.mark.parametrize("orientation", ["LPS", "LAS", "LAI", "RAS", "RPS"])
def test_reorient(nifti_path: str, orientation: str):
    """Tests the reorientation of a nifti image."""
    original_image = nib.load(nifti_path)
    reoriented_image = nifti.reorient(original_image, orientation)

    assert "RAS" == "".join(orientations.aff2axcodes(original_image.affine))
    assert orientation == "".join(orientations.aff2axcodes(reoriented_image.affine))


@pytest.fixture()
def nifti_path(assets_path: str):
    """Path to a nifti test asset file."""
    return os.path.join(assets_path, "vision/datasets/kits23/case_00036/master_00036.nii.gz")
