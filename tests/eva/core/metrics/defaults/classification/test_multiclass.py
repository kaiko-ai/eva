"""MulticlassClassificationMetrics metric tests."""

import pytest
import torch

from eva.core.metrics import defaults

NUM_CLASSES_ONE = 5
PREDS_ONE = torch.tensor(
    [
        [0.70, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.70, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.70, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.70, 0.05],
    ]
)
TARGET_ONE = torch.tensor([0, 1, 3, 2])
EXPECTED_ONE = {
    "MulticlassAccuracy": torch.tensor(0.5000),
    "MulticlassPrecision": torch.tensor(0.5000),
    "MulticlassRecall": torch.tensor(0.5000),
    "MulticlassF1Score": torch.tensor(0.5000),
    "MulticlassAUROC": torch.tensor(0.5333),
}
"""Test features."""


@pytest.mark.parametrize(
    "num_classes, preds, target, expected",
    [
        (NUM_CLASSES_ONE, PREDS_ONE, TARGET_ONE, EXPECTED_ONE),
    ],
)
def test_multiclass_classification_metrics(
    multiclass_classification_metrics: defaults.MulticlassClassificationMetrics,
    preds: torch.Tensor,
    target: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the multiclass_classification_metrics metric."""

    def _calculate_metric() -> None:
        multiclass_classification_metrics.update(preds=preds, target=target)  # type: ignore
        actual = multiclass_classification_metrics.compute()
        torch.testing.assert_close(actual, expected, rtol=1e-04, atol=1e-04)

    _calculate_metric()
    multiclass_classification_metrics.reset()
    _calculate_metric()


def test_multiclass_metrics_filters_ignored_predictions() -> None:
    """Tests that predictions matching ignore_index are filtered before metric update."""
    metrics = defaults.MulticlassClassificationMetrics(
        num_classes=3, input_type="discrete", ignore_index=-1
    )
    # Without filtering, -1 would cause: RuntimeError: bincount only supports
    # 1-d non-negative integral inputs.
    preds = torch.tensor([0, 1, -1, 2])
    target = torch.tensor([0, 1, 2, 2])
    metrics.update(preds=preds, target=target)
    result = metrics.compute()
    # Only the 3 valid samples are evaluated: preds=[0,1,2], target=[0,1,2] -> perfect
    assert result["MulticlassAccuracy"] == 1.0


def test_multiclass_metrics_skips_all_ignored_predictions() -> None:
    """Tests that an all-ignored batch doesn't crash and is effectively skipped."""
    metrics = defaults.MulticlassClassificationMetrics(
        num_classes=3, input_type="discrete", ignore_index=-1
    )
    # First batch: all ignored — should be skipped entirely
    metrics.update(preds=torch.tensor([-1, -1]), target=torch.tensor([0, 1]))
    # Second batch: valid data
    metrics.update(preds=torch.tensor([0, 2]), target=torch.tensor([0, 2]))
    result = metrics.compute()
    # Only second batch counts
    assert result["MulticlassAccuracy"] == 1.0


@pytest.fixture(scope="function")
def multiclass_classification_metrics(num_classes: int) -> defaults.MulticlassClassificationMetrics:
    """MulticlassClassificationMetrics fixture."""
    return defaults.MulticlassClassificationMetrics(num_classes=num_classes)
