"""Shared configuration and fixtures for callbacks unit tests."""

import pytest

from eva.core.data import dataloaders, datamodules, datasets


@pytest.fixture(scope="function")
def datamodule(
    dataset: datasets.TorchDataset,
    dataloader: dataloaders.DataLoader,
) -> datamodules.DataModule:
    """Returns a dummy datamodule fixture."""
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
def dataloader(batch_size: int) -> dataloaders.DataLoader:
    """Test dataloader fixture."""
    return dataloaders.DataLoader(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
