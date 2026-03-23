"""DiceScore metric tests."""

from typing import Tuple

import pytest
import torch

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
        segmentation.DiceScore, batch_size, num_classes, image_size, ignore_index
    )


def test_expected_scores_classwise():
    """Tests the expected score values when calculating class-wise scores."""
    n_samples, n_classes = 4, 3

    target = torch.full((n_samples, n_classes, 128, 128), 0, dtype=torch.int8)
    preds = torch.full((n_samples, n_classes, 128, 128), 0, dtype=torch.int8)
    perfect_preds = preds.clone()

    target[0, 0], perfect_preds[0, 0] = 1, 1
    target[2, 1], perfect_preds[2, 1] = 1, 1

    metric = segmentation.DiceScore(num_classes=n_classes, average="none")
    scores = metric(perfect_preds, target)

    assert scores.tolist() == [1.0] * n_classes

    preds_with_errors = preds.clone()
    preds_with_errors[0, 0] = 1

    scores = metric(preds_with_errors, target)
    assert scores.tolist() == [1.0, 0.75, 1.0]
