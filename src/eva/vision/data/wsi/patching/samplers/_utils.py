from typing import Tuple, List

import numpy as np


def get_grid_coords(
    layer_shape: Tuple[int, int],
    width: int,
    height: int,
    overlap: Tuple[int, int],
    shuffle: bool = True,
    include_last: bool = False,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Get grid coordinates and indices.

    Args:
        layer_shape: The shape of the layer.
        width: The width of the patches.
        height: The height of the patches.
        overlap: The overlap between patches in the grid.
        shuffle: Whether to shuffle the order of the coordinates.
        include_last: Whether to include coordinates of the last patch when it
            it partially exceeds the image and therefore is smaller than the
            specified patch size.
        seed: The random seed.

    Returns:
        A list of tuples with the (x, y) coordinates.
    """
    x_range = range(0, layer_shape[0] - width + 1, width - overlap[0])
    y_range = range(0, layer_shape[1] - height + 1, height - overlap[1])

    x_y = [(x, y) for x in x_range for y in y_range]

    if shuffle:
        indices = list(range(len(x_y)))
        random_generator = np.random.default_rng(seed)
        random_generator.shuffle(indices)
        x_y = [x_y[i] for i in indices]

    return x_y


def validate_dimensions(width: int, height: int, layer_shape: Tuple[int, int]) -> None:
    """Checks if the width / height is bigger than the layer shape.

    Args:
        width: The width of the patches.
        height: The height of the patches.
        layer_shape: The shape of the layer.
    """
    if width > layer_shape[0] or height > layer_shape[1]:
        raise ValueError("The width / height cannot be bigger than the layer shape.")
