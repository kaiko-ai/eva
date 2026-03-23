"""WSI foreground mask tests."""

import os

import numpy as np
import pytest

from eva.vision.data import wsi as eva_wsi

DEFAULT_ARGS = {
    "saturation_threshold": 20,
    "median_blur_kernel_size": 7,
    "fill_holes": False,
    "use_otsu": False,
    "holes_kernel_size": (7, 7),
}


@pytest.mark.parametrize(
    "mask_level_idx, mask_args",
    [
        (0, DEFAULT_ARGS),
        (1, DEFAULT_ARGS),
        (0, DEFAULT_ARGS | {"median_blur_kernel_size": None}),
        (0, DEFAULT_ARGS | {"fill_holes": True}),
        (0, DEFAULT_ARGS | {"use_otsu": True}),
        (0, DEFAULT_ARGS | {"fill_holes": True, "use_otsu": True}),
    ],
)
def test_get_mask(wsi: eva_wsi.Wsi, mask_level_idx: int, mask_args: dict):
    """Tests the foreground mask generation with different configurations."""
    mask = eva_wsi.get_mask(wsi, mask_level_idx=0, **mask_args)

    assert isinstance(mask, eva_wsi.Mask)
    assert isinstance(mask.mask_array, np.ndarray)
    assert mask.mask_array.shape == wsi.level_dimensions[mask.mask_level_idx]
    assert np.all(np.isin(mask.mask_array, [0, 1]))

    if mask.mask_level_idx == 0:
        assert mask.scale_factors == (1.0, 1.0)
    elif mask_level_idx == 1:
        assert mask.scale_factors == (0.5, 0.5)


@pytest.mark.parametrize(
    "width, height, target_mpp, expected_level",
    [
        (4, 4, 0.25, 0),
        (16, 16, 0.05, 0),
        (4, 4, 0.5, 1),
    ],
)
def test_get_mask_level(
    wsi: eva_wsi.Wsi, width: int, height: int, target_mpp: float, expected_level: int
):
    """Tests the selection of the mask level based on the patch dimensions."""
    level = eva_wsi.get_mask_level(wsi, width, height, target_mpp)
    assert level == expected_level


@pytest.mark.parametrize(
    "width, height, target_mpp",
    [
        (4, 4, 0.1),
        (16, 16, 0.01),
        (2, 2, 0.25),
    ],
)
def test_no_suitable_level_available(wsi: eva_wsi.Wsi, width: int, height: int, target_mpp: float):
    """Tests the case where no suitable mask level is available.

    This can happen for instance when the patch dimensions scaled to the selected mask level
    are too small or even collapse to zero pixels.
    """
    with pytest.raises(
        ValueError, match="No level with the specified minimum number of patch pixels available."
    ):
        eva_wsi.get_mask_level(wsi, width, height, target_mpp)


@pytest.fixture
def wsi(assets_path: str) -> eva_wsi.Wsi:
    """Fixture for loading a WSI object.

    The test WSI slide has the following specs:
    - level_dimensions: ((256, 256), (128, 128))
    - level_downsamples: (1.0, 2.0)
    - mpp (level 0): 0.25
    """
    path = os.path.join(assets_path, "vision/datasets/wsi/0/a.tiff")
    return eva_wsi.wsi_backend("openslide")(path)
