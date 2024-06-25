"""MulticlassSegmentationMetrics metric tests."""

import pytest
import torch

from eva.core.metrics import defaults

NUM_CLASSES_ONE = 3
PREDS_ONE = torch.tensor(
    [
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ],
)
TARGET_ONE = torch.tensor(
    [
        [1, 1, 0, 0],
        [1, 1, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ],
    dtype=torch.int32,
)
EXPECTED_ONE = {
    "MulticlassJaccardIndex": torch.tensor(0.4722222089767456),
    "MulticlassF1Score": torch.tensor(0.6222222447395325),
    "MulticlassPrecision": torch.tensor(0.6666666865348816),
    "MulticlassRecall": torch.tensor(0.611011104490899),
}
"""Test features."""


@pytest.mark.parametrize(
    "num_classes, preds, target, expected",
    [
        (NUM_CLASSES_ONE, PREDS_ONE, TARGET_ONE, EXPECTED_ONE),
    ],
)
def test_multiclass_segmentation_metrics(
    multiclass_segmentation_metrics: defaults.MulticlassSegmentationMetrics,
    preds: torch.Tensor,
    target: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the multiclass_segmentation_metrics metric."""

    def _calculate_metric() -> None:
        multiclass_segmentation_metrics.update(preds=preds, target=target)  # type: ignore
        actual = multiclass_segmentation_metrics.compute()
        torch.testing.assert_close(actual, expected, rtol=1e-04, atol=1e-04)

    _calculate_metric()
    multiclass_segmentation_metrics.reset()
    _calculate_metric()


@pytest.fixture(scope="function")
def multiclass_segmentation_metrics(num_classes: int) -> defaults.MulticlassSegmentationMetrics:
    """MulticlassSegmentationMetrics fixture."""
    return defaults.MulticlassSegmentationMetrics(num_classes=num_classes)
