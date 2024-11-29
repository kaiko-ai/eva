"""MulticlassSegmentationMetrics metric tests."""

import pytest
import torch

from eva.vision.metrics import defaults

NUM_BATCHES = 2
BATCH_SIZE = 4
"""Test parameters."""

NUM_CLASSES_ONE = 3
PREDS_ONE = torch.randint(0, NUM_CLASSES_ONE, (NUM_BATCHES, BATCH_SIZE, 32, 32))
TARGET_ONE = torch.randint(0, NUM_CLASSES_ONE, (NUM_BATCHES, BATCH_SIZE, 32, 32))
EXPECTED_ONE = {
    "MonaiDiceScore": torch.tensor(0.34805023670196533),
    "MonaiDiceScore (ignore_empty=False)": torch.tensor(0.34805023670196533),
    "DiceScore (micro)": torch.tensor(0.3482658863067627),
    "DiceScore (macro)": torch.tensor(0.34805023670196533),
    "DiceScore (weighted)": torch.tensor(0.3484232723712921),
    "MeanIoU": torch.tensor(0.2109210342168808),
}
"""Test features."""
assert EXPECTED_ONE["MonaiDiceScore (ignore_empty=False)"] == EXPECTED_ONE["DiceScore (macro)"]


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
        for batch_preds, batch_target in zip(preds, target, strict=False):
            multiclass_segmentation_metrics.update(preds=batch_preds, target=batch_target)  # type: ignore
        actual = multiclass_segmentation_metrics.compute()
        torch.testing.assert_close(actual, expected, rtol=1e-04, atol=1e-04)

    _calculate_metric()
    multiclass_segmentation_metrics.reset()
    _calculate_metric()


@pytest.fixture(scope="function")
def multiclass_segmentation_metrics(num_classes: int) -> defaults.MulticlassSegmentationMetrics:
    """MulticlassSegmentationMetrics fixture."""
    return defaults.MulticlassSegmentationMetrics(num_classes=num_classes)
