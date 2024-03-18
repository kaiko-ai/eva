"""AverageLoss metric tests."""

import pytest
import torch

from eva.core import metrics

BATCH_ONE, BATCH_TWO = [
    torch.tensor(
        [
            [[0.2, 0.1, 3.3], [0.2, 0.1, 3.3], [0.2, 0.1, 3.3]],
            [[0.2, 0.1, 3.3], [0.2, 0.1, 3.3], [0.2, 0.1, 3.3]],
        ]
    ),
    torch.tensor(
        [
            [[0.2, float("nan")], [0.2, 3.3], [float("nan"), 0.1]],
        ]
    ),
]
"""Test features."""


@pytest.mark.parametrize(
    "batch, expected",
    [
        (BATCH_ONE, torch.tensor(1.2)),
        (BATCH_TWO, torch.tensor(float("nan"))),
    ],
)
def test_average_loss_metric(
    average_loss_metric: metrics.AverageLoss,
    batch: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the average_loss_metric metric."""

    def _calculate_metric() -> None:
        for losses in batch:
            average_loss_metric.update(losses)  # type: ignore
        actual = average_loss_metric.compute()
        torch.testing.assert_close(actual, expected, equal_nan=True)

    _calculate_metric()
    average_loss_metric.reset()
    _calculate_metric()


@pytest.fixture(scope="function")
def average_loss_metric() -> metrics.AverageLoss:
    """AverageLoss fixture."""
    return metrics.AverageLoss()
