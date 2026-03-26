"""Tests for the NIfTI I/O functions."""

import os

import nibabel as nib
import numpy as np
import pytest

from eva.vision.utils.io import nifti


@pytest.fixture
def nifti_path(assets_path: str) -> str:
    """Returns the full path to the image file."""
    return os.path.join(assets_path, "vision", "datasets", "btcv", "imagesTr", "img0001.nii.gz")


def test_read_nifti(nifti_path: str) -> None:
    """Tests the read nifti function."""
    nii = nifti.read_nifti(nifti_path)
    assert isinstance(nii, nib.nifti1.Nifti1Image)


def test_nifti_to_array(nifti_path: str) -> None:
    """Tests the nifti to array function."""
    nii = nifti.read_nifti(nifti_path)
    arr = nifti.nifti_to_array(nii)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8, 4)


def test_fetch_shape(nifti_path: str) -> None:
    """Tests the fetch shape function."""
    shape = nifti.fetch_nifti_shape(nifti_path)
    assert shape == (8, 8, 4)


def test_fetch_orientation(nifti_path: str) -> None:
    """Tests the fetch orientation function."""
    orientation = nifti.fetch_nifti_orientation(nifti_path)
    assert orientation.shape == (3, 2)


def test_fetch_axis_code(nifti_path: str) -> None:
    """Tests the fetch axis code function."""
    code = nifti.fetch_nifti_axis_direction_code(nifti_path)
    assert isinstance(code, str)
    assert len(code) == 3
