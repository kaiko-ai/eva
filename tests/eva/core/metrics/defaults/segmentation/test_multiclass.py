"""MulticlassSegmentationMetrics metric tests."""

import pytest
import torch

from eva.core.metrics import defaults

NUM_CLASSES_ONE = 3
NUM_BATCHES = 2
BATCH_SIZE = 4
"""Test parameters."""

PREDS_ONE = torch.randint(0, 3, (2, 4, 32, 32))
TARGET_ONE = torch.randint(0, 3, (2, 4, 32, 32))
EXPECTED_ONE = {
    "GeneralizedDiceScore": torch.tensor(0.34827983379364014),
    "MeanIoU": torch.tensor(0.6965596675872803),
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
