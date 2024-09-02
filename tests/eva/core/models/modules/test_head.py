"""Tests the HeadModule module."""

import math
from typing import Tuple

import pytest
import torch
from torch import nn

from eva.core import metrics, trainers
from eva.core.data import datamodules
from eva.core.models import modules


@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "classification_dataset",
        "classification_dataset_with_metadata",
    ],
)
def test_head_module_fit(
    model: modules.HeadModule,
    datamodule: datamodules.DataModule,
    trainer: trainers.Trainer,
) -> None:
    """Tests the HeadModule fit pipeline."""
    initial_head_weights = model.head.weight.clone()
    trainer.fit(model, datamodule=datamodule)
    # verify that the metrics were updated
    assert trainer.logged_metrics["train/AverageLoss"] > 0
    assert trainer.logged_metrics["val/AverageLoss"] > 0
    # verify that head weights were updated
    assert not torch.all(torch.eq(initial_head_weights, model.head.weight))


@pytest.fixture(scope="function")
def model(
    input_shape: Tuple[int, ...] = (3, 8, 8),
    n_classes: int = 4,
) -> modules.HeadModule:
    """Returns a HeadModule model fixture."""
    return modules.HeadModule(
        head=nn.Linear(math.prod(input_shape), n_classes),
        criterion=nn.CrossEntropyLoss(),
        backbone=nn.Flatten(),
        metrics=metrics.MetricsSchema(
            common=metrics.AverageLoss(),
        ),
    )
