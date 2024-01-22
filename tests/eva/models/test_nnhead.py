"""Tests the NNHead module."""
import math
from typing import Tuple

import pytest
import torch
from torch import nn
from torch.utils import data as torch_data

from eva import metrics, models, trainers
from eva.data import dataloaders, datamodules, datasets


def test_nnhead_fit(
    model: models.NNHead,
    datamodule: datamodules.DataModule,
    trainer: trainers.Trainer,
) -> None:
    """Tests the nnhead fit pipeline."""
    initial_head_weights = model.head.weight.clone()
    trainer.fit(model, datamodule=datamodule)
    assert trainer.logged_metrics["train/AverageLoss"] > 0
    assert trainer.logged_metrics["val/AverageLoss"] > 0
    assert not torch.all(torch.eq(initial_head_weights, model.head.weight))


@pytest.fixture(scope="function")
def model(input_shape: Tuple[int, ...] = (3, 8, 8), n_classes: int = 4) -> models.NNHead:
    """Returns a NNHead model fixture."""
    return models.NNHead(
        head=nn.Linear(math.prod(input_shape), n_classes),
        criterion=nn.CrossEntropyLoss(),
        backbone=nn.Flatten(),
        metrics=metrics.core.MetricsSchema(
            common=metrics.AverageLoss(),
        ),
    )


@pytest.fixture(scope="function")
def datamodule(
    classification_dataset: datasets.Dataset,
    dataloader: dataloaders.DataLoader,
) -> datamodules.DataModule:
    """Returns a dummy classification datamodule fixture."""
    return datamodules.DataModule(
        datasets=datamodules.DatasetsSchema(
            train=classification_dataset,
            val=classification_dataset,
        ),
        dataloaders=datamodules.DataloadersSchema(
            train=dataloader,
            val=dataloader,
        ),
    )


@pytest.fixture(scope="function")
def trainer(max_epochs: int = 1) -> trainers.Trainer:
    """Returns a model trainer fixture."""
    return trainers.Trainer(max_epochs=max_epochs, accelerator="cpu")


@pytest.fixture(scope="function")
def classification_dataset(
    n_samples: int = 4,
    input_shape: Tuple[int, ...] = (3, 8, 8),
    target_shape: Tuple[int, ...] = (),
    n_classes: int = 4,
) -> datasets.Dataset:
    """Dummy classification dataset fixture."""
    return torch_data.TensorDataset(
        torch.randn((n_samples,) + input_shape),
        torch.randint(n_classes, (n_samples,) + target_shape, dtype=torch.long),
    )


@pytest.fixture(scope="function")
def dataloader(batch_size: int = 1) -> dataloaders.DataLoader:
    """Test dataloader fixture."""
    return dataloaders.DataLoader(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
