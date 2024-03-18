"""Tests the core dataloader."""

import pytest
from torch.utils import data as torch_data
from typing_extensions import override

from eva.core.data import dataloaders, datasets


@pytest.mark.parametrize(
    "batch_size, shuffle, num_workers, pin_memory",
    [
        (16, True, 4, True),
        (4, False, 8, False),
        (None, True, 1, False),
    ],
)
def test_dataloader(
    dataloader: dataloaders.DataLoader,
    dataset: datasets.Dataset,
) -> None:
    """Tests the core dataloader."""
    torch_dataloader = dataloader(dataset)
    assert torch_dataloader.batch_size == dataloader.batch_size
    assert torch_dataloader.num_workers == dataloader.num_workers
    assert torch_dataloader.pin_memory == dataloader.pin_memory
    assert isinstance(torch_dataloader, torch_data.DataLoader)


@pytest.fixture(scope="function")
def dataloader(
    batch_size: int | None,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> dataloaders.DataLoader:
    """Dataloader fixture."""
    return dataloaders.DataLoader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@pytest.fixture(scope="function")
def dataset() -> datasets.Dataset:
    """A dummy dataset fixture."""

    class DummyDataset(datasets.Dataset):
        """Dummy dataset."""

        @override
        def __getitem__(self, index: int) -> int:
            return 1

        def __len__(self) -> int:
            """Returns total number of samples of the dataset."""
            return 100

    return DummyDataset()
