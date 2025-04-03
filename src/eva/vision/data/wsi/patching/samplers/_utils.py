from typing import List, Tuple

import numpy as np


def get_grid_coords(
    image_size: Tuple[int, int],
    width: int,
    height: int,
    overlap: Tuple[int, int],
    shuffle: bool = True,
    include_partial_patches: bool = False,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Get grid coordinates and indices.

    Args:
        image_size: The shape of the complete image.
        width: The width of the patches.
        height: The height of the patches.
        overlap: The overlap between patches in the grid.
        shuffle: Whether to shuffle the order of the coordinates.
        include_partial_patches: Whether to include coordinates of the last patch when it
            it partially exceeds the image and therefore is smaller than the
            specified patch size.
        seed: The random seed.

    Returns:
        A list of tuples with the (x, y) coordinates.
    """
    if not include_partial_patches and (width > image_size[0] or height > image_size[1]):
        raise ValueError("The patch size cannot be bigger than the image.")

    x_stop = image_size[0] if include_partial_patches else image_size[0] - width + 1
    y_stop = image_size[1] if include_partial_patches else image_size[1] - height + 1

    x_coords = list(range(0, x_stop, width - overlap[0])) or [0]
    y_coords = list(range(0, y_stop, height - overlap[1])) or [0]

    x_y = [(x, y) for x in x_coords for y in y_coords]

    if shuffle:
        random_generator = np.random.default_rng(seed)
        random_generator.shuffle(x_y)

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
