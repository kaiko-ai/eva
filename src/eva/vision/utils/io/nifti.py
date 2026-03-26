"""NIfTI I/O related functions."""

import abc
from dataclasses import dataclass
from typing import Any, Tuple

import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel import orientations
from typing_extensions import override

from eva.core.utils.suppress_logs import SuppressLogs
from eva.vision.utils.io import _utils


class Sampler(abc.ABC):
    """Base class for slice sampling strategies."""

    @abc.abstractmethod
    def sample(self, total: int) -> list[int]:
        """Return sampled slice indices."""


@dataclass
class IndexSampler(Sampler):
    """Sample specific slice indices."""

    indices: list[int]
    """List of indices to sample."""

    @override
    def sample(self, total: int) -> list[int]:
        return self.indices


@dataclass
class BlockSampler(Sampler):
    """Sample a contiguous block of N slices."""

    n: int
    """The maximum number of slices to sample."""

    @override
    def sample(self, total: int) -> list[int]:
        k = min(total, self.n)
        start = np.random.randint(0, total - k + 1)
        return list(range(start, start + k))


@dataclass
class UniformSampler(Sampler):
    """Sample N slices uniformly across the scan."""

    n: int
    """The maximum number of slices to sample."""

    @override
    def sample(self, total: int) -> list[int]:
        return np.linspace(0, total - 1, min(total, self.n), dtype=int).tolist()


@dataclass
class GaussianSampler(Sampler):
    """Sample N slices according to a Gaussian distribution."""

    n: int
    """The maximum number of slices to sample."""

    mean: float | None = None
    """Center of the Gaussian in slice coordinates. Defaults to the middle."""

    std: float | None = None
    """Standard deviation of the Gaussian. Defaults to total / 3."""

    @override
    def sample(self, total: int) -> list[int]:
        """Return sampled slice indices."""
        samples = np.random.normal(
            loc=self.mean if self.mean is not None else (total - 1) / 2,
            scale=self.std if self.std is not None else total / 3,
            size=min(total, self.n),
        )
        indices = np.clip(np.round(samples), 0, total - 1).astype(int)
        return np.sort(indices).tolist()


def read_nifti(
    path: str,
    *,
    sampler: Sampler | None = None,
    orientation: str | None = None,
    orientation_reference: str | None = None,
) -> nib.nifti1.Nifti1Image:
    """Reads and loads a NIfTI image from a file path.

    Args:
        path: The path to the NIfTI file.
        sampler: Strategy for sampling slices. Supports:
            - IndexSampler(indices=[...]): Specific slice indices.
            - BlockSampler(n=...): A contiguous block of N slices.
            - UniformSampler(n=...): N slices sampled uniformly.
            - GaussianSampler(n=...): N Gaussian sampled slices.
        orientation: The orientation code to reorient the nifti image.
        orientation_reference: Path to a NIfTI file which
            will be used as a reference for the orientation
            transform in case the file missing the pixdim array
            in the NIfTI header.

    Returns:
        The NIfTI image class instance.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    image_data = _load_nifti_silently(path)

    if sampler:
        total = image_data.shape[-1]
        slice_indices = sampler.sample(total)
        proxy_slices = [image_data.dataobj[:, :, i] for i in slice_indices]
        image_data = nib.nifti1.Nifti1Image(
            np.stack(proxy_slices, axis=-1),
            image_data.affine,
            image_data.header,
        )

    if orientation:
        image_data = _reorient(
            image_data, orientation=orientation, reference_file=orientation_reference
        )

    return image_data


def nifti_to_array(
    nii: nib.nifti1.Nifti1Image,
    /,
    *,
    dtype: np.dtype | type | None = np.int16,
) -> npt.NDArray[Any]:
    """Converts a NIfTI image to a numpy array with efficient casting.

    Args:
        nii: The input NIfTI image.
        dtype: The type to cast the data to. If `None`, it will
            cast the raw image array to the inferred type via
            `use_storage_dtype`.

    Returns:
        The image as a numpy array (height, width, channels).
    """
    return np.array(nii.dataobj, dtype=dtype or nii.get_data_dtype(), order="C", copy=True)


def save_array_as_nifti(
    array: npt.ArrayLike,
    filename: str,
    *,
    dtype: npt.DTypeLike | None = np.int64,
) -> None:
    """Saved a numpy array as a NIfTI image file.

    Args:
        array: The image array to save.
        filename: The name to save the image like.
        dtype: The data type to save the image.
    """
    nifti_image = nib.nifti1.Nifti1Image(array, affine=np.eye(4), dtype=dtype)
    nifti_image.to_filename(filename)


def fetch_nifti_shape(path: str) -> Tuple[int]:
    """Fetches the NIfTI image shape from a file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The image shape.

    Raises:
        FileExistsError: If the path does not exist or it is unreachable.
        ValueError: If the input channel is invalid for the image.
    """
    _utils.check_file(path)
    nii = _load_nifti_silently(path)
    return nii.header.get_data_shape()  # type: ignore


def fetch_nifti_orientation(path: str) -> npt.NDArray[Any]:
    """Fetches the NIfTI image orientation.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The array orientation.
    """
    _utils.check_file(path)
    nii = _load_nifti_silently(path)
    return nib.orientations.io_orientation(nii.affine)


def fetch_nifti_axis_direction_code(path: str) -> str:
    """Fetches the NIfTI axis direction code from a file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The axis direction codes as string (e.g. "LAS").
    """
    _utils.check_file(path)
    image_data = nib.nifti1.load(path)
    return "".join(orientations.aff2axcodes(image_data.affine))


def _load_nifti_silently(path: str) -> nib.nifti1.Nifti1Image:
    """Reads a NIfTI image in silent mode."""
    with SuppressLogs():
        return nib.nifti1.load(path)  # type: ignore
    raise ValueError(f"Failed to load NIfTI file: {path}")


def _reorient(
    nii: nib.nifti1.Nifti1Image,
    /,
    orientation: str | tuple[str, str, str] = "RAS",
    reference_file: str | None = None,
) -> nib.nifti1.Nifti1Image:
    """Reorients a NIfTI image to a specified orientation.

    Args:
        nii: The input NIfTI image.
        orientation: Desired orientation expressed as a
            three-character string (e.g., "RAS") or a tuple
            (e.g., ("R", "A", "S")).
        reference_file: Path to a reference NIfTI file whose
            orientation should be used if the input image lacks
            a valid affine transformation.

    Returns:
        The reoriented NIfTI image.
    """
    affine_matrix, _ = nii.get_qform(coded=True)
    orig_ornt = (
        fetch_nifti_orientation(reference_file)
        if reference_file and affine_matrix is None
        else nib.orientations.io_orientation(nii.affine)
    )
    targ_ornt = orientations.axcodes2ornt(orientation)
    transform = orientations.ornt_transform(orig_ornt, targ_ornt)
    return nii.as_reoriented(transform)  # type: ignore
