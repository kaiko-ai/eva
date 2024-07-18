"""Tests the BatchPostProcess module."""

from typing import Any, Dict, List

import pytest
import torch
from torch import nn

from eva.core.models.modules.utils import batch_postprocess

BATCH_OUTPUTS_1 = {
    "predictions": torch.Tensor([1.5, -0.3]),
    "targets": torch.Tensor([1, 1]),
}
EXPECTED_11 = {
    "predictions": torch.Tensor([0.8176, 0.4256]),
    "targets": torch.Tensor([1, 1]),
}
EXPECTED_12 = {
    "predictions": torch.Tensor([0.6937, 0.6048]),
    "targets": torch.Tensor([1, 1]),
}
EXPECTED_13 = {
    "predictions": torch.Tensor([1.5, -0.3]),
    "targets": torch.Tensor([0, 0]),
}
EXPECTED_14 = {
    "predictions": torch.Tensor([0.8176, 0.4256]),
    "targets": torch.Tensor([0, 0]),
}
"""Test inputs and assets."""


@pytest.mark.parametrize(
    "targets_transforms, predictions_transforms, outputs, expected",
    [
        (None, [nn.Sigmoid()], BATCH_OUTPUTS_1, EXPECTED_11),
        (None, [nn.Sigmoid(), nn.Sigmoid()], BATCH_OUTPUTS_1, EXPECTED_12),
        ([lambda tensor: tensor - 1], None, BATCH_OUTPUTS_1, EXPECTED_13),
        ([lambda tensor: tensor - 1], [nn.Sigmoid()], BATCH_OUTPUTS_1, EXPECTED_14),
    ],
)
def test_batch_postprocess_call(
    processes: batch_postprocess.Transform,
    outputs: Dict[str, torch.Tensor],
    expected: Dict[str, torch.Tensor],
) -> None:
    """Tests the BatchPostProcess `__call__` method."""
    _outputs = {key: torch.clone(value) for key, value in outputs.items()}
    processes(_outputs)  # type: ignore
    torch.testing.assert_close(expected, _outputs, rtol=1e-4, atol=1e-4)


@pytest.fixture(scope="function")
def processes(
    targets_transforms: List[batch_postprocess.Transform | Dict[str, Any]] | None,
    predictions_transforms: List[batch_postprocess.Transform | Dict[str, Any]] | None,
) -> batch_postprocess.BatchPostProcess:
    """Returns a BatchPostProcess fixture."""
    return batch_postprocess.BatchPostProcess(
        predictions_transforms=predictions_transforms,
        targets_transforms=targets_transforms,
    )
