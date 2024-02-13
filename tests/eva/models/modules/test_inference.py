"""Tests the HeadModule module."""

import math
from typing import Tuple

import pytest
import torch
from torch import nn

from eva import trainers
from eva.data import datamodules
from eva.models import modules

N_CLASSES = 4


@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "classification_dataset",
        "classification_dataset_with_metadata",
    ],
)
def test_inference_module_predict(
    model: modules.InferenceModule,
    datamodule: datamodules.DataModule,
    trainer: trainers.Trainer,
) -> None:
    """Tests the HeadModule fit pipeline."""
    predictions = trainer.predict(model, datamodule=datamodule)
    dataset_size = len(datamodule.datasets.predict)  # type: ignore

    assert isinstance(datamodule.dataloaders.predict.batch_size, int)
    assert isinstance(predictions, list)
    n_batches = math.ceil(dataset_size / datamodule.dataloaders.predict.batch_size)
    for p in predictions:
        assert isinstance(p, torch.Tensor)
        assert p.shape[0] == n_batches
        assert p.shape[1] == N_CLASSES


@pytest.fixture(scope="function")
def model(
    input_shape: Tuple[int, ...] = (3, 8, 8), n_classes: int = N_CLASSES
) -> modules.InferenceModule:
    """Returns a HeadModule model fixture."""
    return modules.InferenceModule(
        model=nn.Sequential(nn.Flatten(), nn.Linear(math.prod(input_shape), n_classes))
    )
