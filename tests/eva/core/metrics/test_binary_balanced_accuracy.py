"""AverageLoss metric tests."""

import pytest
import torch

from eva.core import metrics

PREDS_ONE = torch.tensor([[0.70, 0.95, 0.19, 0.10, 0.59]])
TARGET_ONE = torch.tensor([[1, 1, 0, 0, 0]])
"""Test features one."""

PREDS_TWO = torch.tensor(
    [
        [0.70, 0.95, 0.19, 0.10, 0.59],
        [0.70, 0.95, 0.19, 0.10, 0.59],
    ],
)
TARGET_TWO = torch.tensor([[0, 1, 1, 0, 1], [0, 1, 1, 0, 1]])
"""Test features two."""


@pytest.mark.parametrize(
    "preds, target, expected",
    [
        (PREDS_ONE, TARGET_ONE, torch.tensor(0.8333333730697632)),
        (PREDS_TWO, TARGET_TWO, torch.tensor(0.5833333730697632)),
    ],
)
def test_binary_balanced_accuracy_metric(
    binary_balanced_accuracy_metric: metrics.BinaryBalancedAccuracy,
    preds: torch.Tensor,
    target: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the binary_balanced_accuracy_metric metric."""

    def _calculate_metric() -> None:
        for batch_preds, batch_target in zip(preds, target, strict=False):
            binary_balanced_accuracy_metric.update(batch_preds, batch_target)  # type: ignore
        actual = binary_balanced_accuracy_metric.compute()
        torch.testing.assert_close(actual, expected, equal_nan=True)

    _calculate_metric()
    binary_balanced_accuracy_metric.reset()
    _calculate_metric()


@pytest.fixture(scope="function")
def binary_balanced_accuracy_metric() -> metrics.BinaryBalancedAccuracy:
    """BinaryBalancedAccuracy fixture."""
    return metrics.BinaryBalancedAccuracy()
