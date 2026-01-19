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


@pytest.mark.parametrize(
    "compute_groups, separator",
    [
        (True, "/"),
        (False, "/"),
        (True, "_"),
        (False, "-"),
    ],
)
def test_metric_module_schema_options(compute_groups: bool, separator: str) -> None:
    """Tests that compute_groups and separator are correctly passed from schema."""
    schema = structs.MetricsSchema(
        common=torchmetrics.Accuracy(task="binary"),
        compute_groups=compute_groups,
        separator=separator,
    )
    metric_module = structs.MetricModule.from_schema(schema=schema)

    # Check that separator is applied in metric prefixes
    for name in metric_module.training_metrics.keys():
        assert str(name).startswith(f"train{separator}")
    for name in metric_module.validation_metrics.keys():
        assert str(name).startswith(f"val{separator}")
    for name in metric_module.test_metrics.keys():
        assert str(name).startswith(f"test{separator}")
