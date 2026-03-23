from typing import Any, Tuple

import numpy.typing as npt
import torch
from torchvision import tv_tensors

from eva.vision.data import tv_tensors as eva_tv_tensors
from eva.vision.data.datasets import wsi
from eva.vision.utils import io


def get_coords_at_index(
    dataset: wsi.MultiWsiDataset, index: int
) -> Tuple[Tuple[int, int], int, int]:
    """Returns the coordinates ((x,y),width,height) of the patch at the given index.

    Args:
        dataset: The WSI dataset instance.
        index: The sample index.
    """
    image_index = dataset._get_dataset_idx(index)
    patch_index = index if image_index == 0 else index - dataset.cumulative_sizes[image_index - 1]
    wsi_dataset = dataset.datasets[image_index]
    if isinstance(wsi_dataset, wsi.WsiDataset):
        coords = wsi_dataset._coords
        return coords.x_y[patch_index], coords.width, coords.height
    else:
        raise Exception(f"Expected WsiDataset, got {type(wsi_dataset)}")


def extract_mask_patch(
    mask: npt.NDArray[Any], dataset: wsi.MultiWsiDataset, index: int
) -> npt.NDArray[Any]:
    """Reads the mask patch at the coordinates corresponding to the dataset index.

    Args:
        mask: The mask array.
        dataset: The WSI dataset instance.
        index: The sample index.
    """
    (x, y), width, height = get_coords_at_index(dataset, index)
    return mask[y : y + height, x : x + width]


def load_volume_tensor(file: str, orientation: str = "PLS") -> eva_tv_tensors.Volume:
    """Load a volume from NIfTI file as :class:`eva.vision.data.tv_tensors.Volume`.

    Args:
        file: The path to the NIfTI file.
        orientation: The orientation code to reorient the nifti image.

    Returns:
        Volume tensor representing of shape `[T, C, H, W]`.
    """
    nii = io.read_nifti(file, orientation=orientation)
    array = io.nifti_to_array(nii)
    array_reshaped_tchw = array[None, :, :, :].transpose(3, 0, 1, 2)

    if nii.affine is None:
        raise ValueError(f"Affine matrix is missing for {file}.")
    affine = torch.tensor(nii.affine[:, [2, 0, 1, 3]], dtype=torch.float32)

    return eva_tv_tensors.Volume(
        array_reshaped_tchw, affine=affine, dtype=torch.float32
    )  # type: ignore


def load_mask_tensor(
    file: str, volume_file: str | None = None, orientation: str = "PLS"
) -> tv_tensors.Mask:
    """Load a volume from NIfTI file as :class:`torchvision.tv_tensors.Mask`.

    Args:
        file: The path to the NIfTI file containing the mask.
        volume_file: The path to the volume file used as orientation reference in case
            the mask file is missing the pixdim array in the NIfTI header.
        orientation: The orientation code to reorient the nifti image.

    Returns:
        Mask tensor of shape `[T, C, H, W]`.
    """
    nii = io.read_nifti(file, orientation="PLS", orientation_reference=volume_file)
    array = io.nifti_to_array(nii)
    array_reshaped_tchw = array[None, :, :, :].transpose(3, 0, 1, 2)
    return tv_tensors.Mask(array_reshaped_tchw, dtype=torch.long)  # type: ignore
