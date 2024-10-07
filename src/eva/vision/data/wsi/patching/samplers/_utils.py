from typing import Tuple

import numpy as np


def get_grid_coords_and_indices(
    layer_shape: Tuple[int, int],
    width: int,
    height: int,
    overlap: Tuple[int, int],
    shuffle: bool = True,
    seed: int = 42,
):
    """Get grid coordinates and indices.

    Args:
        layer_shape: The shape of the layer.
        width: The width of the patches.
        height: The height of the patches.
        overlap: The overlap between patches in the grid.
        shuffle: Whether to shuffle the indices.
        seed: The random seed.
    """
    x_range = range(0, layer_shape[0] - width + 1, width - overlap[0])
    y_range = range(0, layer_shape[1] - height + 1, height - overlap[1])
    x_y = [(x, y) for x in x_range for y in y_range]

    indices = list(range(len(x_y)))
    if shuffle:
        random_generator = np.random.default_rng(seed)
        random_generator.shuffle(indices)
    return x_y, indices


def validate_dimensions(width: int, height: int, layer_shape: Tuple[int, int]) -> None:
    """Checks if the width / height is bigger than the layer shape.

    Args:
        width: The width of the patches.
        height: The height of the patches.
        layer_shape: The shape of the layer.
    """
    if width > layer_shape[0] or height > layer_shape[1]:
        raise ValueError("The width / height cannot be bigger than the layer shape.")
