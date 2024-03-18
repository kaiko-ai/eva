"""Shared configuration and fixtures for models/modules unit tests."""

from typing import Tuple

import pytest
import torch
from torch.utils import data as torch_data

from eva.core.data import dataloaders, datamodules, datasets
from eva.core.trainers import trainer as eva_trainer


@pytest.fixture(scope="function")
def datamodule(
    request: pytest.FixtureRequest,
    dataset_fixture: str,
    dataloader: dataloaders.DataLoader,
) -> datamodules.DataModule:
    """Returns a dummy datamodule fixture."""
    dataset = request.getfixturevalue(dataset_fixture)
    return datamodules.DataModule(
        datasets=datamodules.DatasetsSchema(
            train=dataset,
            val=dataset,
            predict=dataset,
        ),
        dataloaders=datamodules.DataloadersSchema(
            train=dataloader,
            val=dataloader,
            predict=dataloader,
        ),
    )


@pytest.fixture(scope="function")
def trainer(max_epochs: int = 1) -> eva_trainer.Trainer:
    """Returns a model trainer fixture."""
    return eva_trainer.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        default_root_dir="logs/test",
    )


@pytest.fixture(scope="function")
def classification_dataset(
    n_samples: int = 4,
    input_shape: Tuple[int, ...] = (3, 8, 8),
    target_shape: Tuple[int, ...] = (),
    n_classes: int = 4,
) -> datasets.TorchDataset:
    """Dummy classification dataset fixture."""
    return torch_data.TensorDataset(
        torch.randn((n_samples,) + input_shape),
        torch.randint(n_classes, (n_samples,) + target_shape, dtype=torch.long),
    )


@pytest.fixture(scope="function")
def classification_dataset_with_metadata(
    n_samples: int = 4,
    input_shape: Tuple[int, ...] = (3, 8, 8),
    target_shape: Tuple[int, ...] = (),
    n_classes: int = 4,
) -> datasets.TorchDataset:
    """Dummy classification dataset fixture with metadata."""
    return torch_data.TensorDataset(
        torch.randn((n_samples,) + input_shape),
        torch.randint(n_classes, (n_samples,) + target_shape, dtype=torch.long),
        torch.randint(2, (n_samples,) + target_shape, dtype=torch.long),
    )


@pytest.fixture(scope="function")
def dataloader(batch_size: int = 2) -> dataloaders.DataLoader:
    """Test dataloader fixture."""
    return dataloaders.DataLoader(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
