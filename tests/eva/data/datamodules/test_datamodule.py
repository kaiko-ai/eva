"""Tests for the core datamodule."""

from typing import List

import pytest
from torch.utils import data as torch_data

from eva.data import dataloaders, datamodules, datasets
from eva.data.datamodules import schemas
from tests.eva.data.datamodules import _utils


def test_datamodule_methods(datamodule: datamodules.DataModule) -> None:
    """Tests the core datamodule methods."""

    def assert_dataset(dataset: datasets.Dataset | List[datasets.Dataset] | None):
        assert isinstance(dataset, _utils.DummyDataset)
        assert dataset._prepare_data_called is True
        assert dataset._setup_called is True
        assert dataset._teardown_called is True

    datamodule.prepare_data()
    datamodule.setup(stage="train")
    datamodule.teardown(stage="train")

    assert_dataset(datamodule.datasets.train)
    assert_dataset(datamodule.datasets.val)
    assert_dataset(datamodule.datasets.test)
    assert_dataset(datamodule.datasets.predict)


def test_datamodule_dataloaders(datamodule: datamodules.DataModule) -> None:
    """Tests the core datamodule dataloaders."""

    def _assert_evaluation_dataloader(dataloaders: List[torch_data.DataLoader]) -> None:
        """Asserts the evaluation dataloaders."""
        assert isinstance(dataloaders, list) and isinstance(dataloaders[0], torch_data.DataLoader)

    assert isinstance(datamodule.train_dataloader(), torch_data.DataLoader)
    _assert_evaluation_dataloader(datamodule.val_dataloader())
    _assert_evaluation_dataloader(datamodule.test_dataloader())
    _assert_evaluation_dataloader(datamodule.predict_dataloader())


@pytest.fixture(scope="function")
def datamodule() -> datamodules.DataModule:
    """DataModule fixture."""
    return datamodules.DataModule(
        datasets=schemas.DatasetsSchema(
            train=_utils.DummyDataset(),
            val=_utils.DummyDataset(),
            test=_utils.DummyDataset(),
            predict=_utils.DummyDataset(),
        ),
        dataloaders=schemas.DataloadersSchema(
            train=dataloaders.DataLoader(),
            val=dataloaders.DataLoader(),
            test=dataloaders.DataLoader(),
            predict=dataloaders.DataLoader(),
        ),
    )
