"""MetricsSchema tests."""

import pytest
import torchmetrics

from eva.core.metrics import structs
from eva.core.metrics.structs.typings import MetricModuleType


@pytest.mark.parametrize(
    "common, train, evaluation, expected_train, expected_evaluation",
    [
        (
            torchmetrics.Accuracy("binary"),
            None,
            None,
            "BinaryAccuracy()",
            "BinaryAccuracy()",
        ),
        (
            None,
            torchmetrics.Accuracy("binary"),
            None,
            "BinaryAccuracy()",
            "None",
        ),
        (
            None,
            None,
            torchmetrics.Accuracy("binary"),
            "None",
            "BinaryAccuracy()",
        ),
        (
            torchmetrics.Accuracy("binary"),
            torchmetrics.Dice(),
            None,
            "[BinaryAccuracy(), Dice()]",
            "BinaryAccuracy()",
        ),
        (
            torchmetrics.Accuracy("binary"),
            None,
            torchmetrics.Dice(),
            "BinaryAccuracy()",
            "[BinaryAccuracy(), Dice()]",
        ),
        (
            torchmetrics.Accuracy("binary"),
            torchmetrics.Dice(),
            torchmetrics.AUROC("binary"),
            "[BinaryAccuracy(), Dice()]",
            "[BinaryAccuracy(), BinaryAUROC()]",
        ),
    ],
)
def test_metric_schema(
    metric_schema: structs.MetricsSchema,
    expected_train: str,
    expected_evaluation: str,
):
    """Tests MetricsSchema."""
    assert str(metric_schema.training_metrics) == expected_train
    assert str(metric_schema.evaluation_metrics) == expected_evaluation


@pytest.fixture(scope="function")
def metric_schema(
    common: MetricModuleType | None,
    train: MetricModuleType | None,
    evaluation: MetricModuleType | None,
) -> structs.MetricsSchema:
    """MetricsSchema fixture."""
    return structs.MetricsSchema(
        common=common,
        train=train,
        evaluation=evaluation,
    )
