"""MetricModule tests."""

from typing import List

import pytest
import torchmetrics

from eva.core.metrics import structs


@pytest.mark.parametrize(
    "schema, expected",
    [
        (structs.MetricsSchema(train=torchmetrics.Dice()), [1, 0, 0]),
        (structs.MetricsSchema(evaluation=torchmetrics.Dice()), [0, 1, 1]),
        (structs.MetricsSchema(common=torchmetrics.Dice()), [1, 1, 1]),
        (
            structs.MetricsSchema(train=torchmetrics.Dice(), evaluation=torchmetrics.Dice()),
            [1, 1, 1],
        ),
    ],
)
def test_metric_module(metric_module: structs.MetricModule, expected: List[int]) -> None:
    """Tests the MetricModule."""
    assert len(metric_module.training_metrics) == expected[0]
    assert len(metric_module.validation_metrics) == expected[1]
    assert len(metric_module.test_metrics) == expected[2]
    # test that the metrics are copied and are not the same object
    assert metric_module.training_metrics != metric_module.validation_metrics
    assert metric_module.training_metrics != metric_module.test_metrics
    assert metric_module.validation_metrics != metric_module.test_metrics


@pytest.fixture(scope="function")
def metric_module(schema: structs.MetricsSchema) -> structs.MetricModule:
    """MetricModule fixture."""
    return structs.MetricModule.from_schema(schema=schema)
