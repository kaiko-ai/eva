"""ModelModule metric tests."""
import pytest
import torch
from eva import metrics as metrics_lib
from eva import models


def test_model_module(model_module: models.ModelModule) -> None:
    """"""
    model_module.on_train_batch_end(
        outputs={
            "loss": torch.rand(1)[0],
            "preds": torch.tensor([0, 1, 0]),
            "target": torch.tensor([0, 1, 1]),
        },
        batch=(torch.rand(3), torch.rand(3)),
        batch_idx=0,
    )


@pytest.fixture(scope="function")
def model_module() -> models.ModelModule:
    """AverageLoss fixture."""
    return models.ModelModule(
        metrics=metrics_lib.core.MetricsSchema(
            common=metrics_lib.AverageLoss(),
        )
    )
