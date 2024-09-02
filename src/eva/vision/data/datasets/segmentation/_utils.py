from typing import Any, Tuple

import numpy.typing as npt

from eva.vision.data.datasets import wsi


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
