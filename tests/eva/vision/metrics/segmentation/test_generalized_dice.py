"""GeneralizedDiceScore metric tests."""

from typing import Tuple

import pytest

from eva.vision.metrics import segmentation
from tests.eva.vision.metrics.segmentation import _utils


@pytest.mark.parametrize(
    "batch_size, num_classes, image_size, ignore_index",
    [
        (4, 3, (16, 16), 0),
        (16, 5, (20, 20), 1),
    ],
)
def test_ignore_index(
    batch_size: int, num_classes: int, image_size: Tuple[int, int], ignore_index: int
) -> None:
    """Tests the `ignore_index` functionality."""
    _utils._test_ignore_index(
        segmentation.GeneralizedDiceScore, batch_size, num_classes, image_size, ignore_index
    )
