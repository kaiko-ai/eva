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


@pytest.fixture(scope="function")
def multiclass_classification_metrics(num_classes: int) -> defaults.MulticlassClassificationMetrics:
    """MulticlassClassificationMetrics fixture."""
    return defaults.MulticlassClassificationMetrics(num_classes=num_classes)
