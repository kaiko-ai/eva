"""BinaryClassificationMetrics metric tests."""

import pytest
import torch

from eva.core.metrics import defaults

PREDS_ONE = torch.tensor([0.70, 0.05, 0.99, 0.10, 0.3])
TARGET_ONE = torch.tensor([0, 1, 1, 0, 1])
EXPECTED_ONE = {
    "BinaryAUROC": torch.tensor(0.5000),
    "BinaryAccuracy": torch.tensor(0.4000),
    "BinaryBalancedAccuracy": torch.tensor(0.4166),
    "BinaryF1Score": torch.tensor(0.4000),
    "BinaryPrecision": torch.tensor(0.5000),
    "BinaryRecall": torch.tensor(0.3333),
}
"""Test features."""


@pytest.mark.parametrize(
    "preds, target, expected",
    [
        (PREDS_ONE, TARGET_ONE, EXPECTED_ONE),
    ],
)
def test_binary_classification_metrics(
    binary_classification_metrics: defaults.BinaryClassificationMetrics,
    preds: torch.Tensor,
    target: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the binary_classification_metrics metric."""

    def _calculate_metric() -> None:
        binary_classification_metrics.update(preds=preds, target=target)  # type: ignore
        actual = binary_classification_metrics.compute()
        torch.testing.assert_close(actual, expected, rtol=1e-04, atol=1e-04)

    _calculate_metric()
    binary_classification_metrics.reset()
    _calculate_metric()


@pytest.fixture(scope="function")
def binary_classification_metrics() -> defaults.BinaryClassificationMetrics:
    """BinaryClassificationMetrics fixture."""
    return defaults.BinaryClassificationMetrics()
