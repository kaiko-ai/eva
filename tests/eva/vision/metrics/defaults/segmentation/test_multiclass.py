"""MulticlassSegmentationMetrics metric tests."""

import pytest
import torch

from eva.core.metrics import structs
from eva.vision.metrics import defaults

NUM_BATCHES = 2
BATCH_SIZE = 4
NUM_CLASSES = 3
PREDS = torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32))
TARGET = torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32))
EXPECTED_ONE = {
    "MonaiDiceScore": torch.tensor(0.34805023670196533),
    "MonaiDiceScore (ignore_empty=False)": torch.tensor(0.34805023670196533),
    "DiceScore (micro)": torch.tensor(0.3482658863067627),
    "DiceScore (macro)": torch.tensor(0.34805023670196533),
    "DiceScore (weighted)": torch.tensor(0.3484232723712921),
    "MeanIoU": torch.tensor(0.2109210342168808),
}
assert EXPECTED_ONE["MonaiDiceScore (ignore_empty=False)"] == EXPECTED_ONE["DiceScore (macro)"]
PREDS_ONE_HOT = torch.nn.functional.one_hot(PREDS, num_classes=NUM_CLASSES).movedim(-1, -3)
TARGET_ONE_HOT = torch.nn.functional.one_hot(TARGET, num_classes=NUM_CLASSES).movedim(-1, -3)
EXPECTED_TWO = {
    "DiceScore (macro)": torch.tensor(0.34805023670196533),
    "DiceScore (macro/global)": torch.tensor(0.3483232259750366),
}


@pytest.mark.parametrize(
    "metrics_collection, preds, target, expected",
    [
        ("multiclass_segmentation_metrics", PREDS, TARGET, EXPECTED_ONE),
        ("multiclass_segmentation_metrics_v2", PREDS_ONE_HOT, TARGET_ONE_HOT, EXPECTED_TWO),
    ],
    indirect=["metrics_collection"],
)
def test_multiclass_segmentation_metrics(
    metrics_collection: structs.MetricCollection,
    preds: torch.Tensor,
    target: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Tests the multiclass_segmentation_metrics metric."""

    def _calculate_metric() -> None:
        for batch_preds, batch_target in zip(preds, target, strict=False):
            metrics_collection.update(preds=batch_preds, target=batch_target)  # type: ignore
        actual = metrics_collection.compute()
        torch.testing.assert_close(actual, expected, rtol=1e-04, atol=1e-04)

    _calculate_metric()
    metrics_collection.reset()
    _calculate_metric()


@pytest.fixture(scope="function")
def metrics_collection(request) -> structs.MetricCollection:
    """Indirect fixture that returns the appropriate metrics class."""
    if request.param == "multiclass_segmentation_metrics":
        return defaults.MulticlassSegmentationMetrics(num_classes=NUM_CLASSES)
    elif request.param == "multiclass_segmentation_metrics_v2":
        return defaults.MulticlassSegmentationMetricsV2(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown metrics fixture: {request.param}")
