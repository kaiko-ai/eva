"""Tests for the core datamodule."""

from typing import List

import pytest
from torch.utils import data as torch_data

from eva.core.data import dataloaders, datamodules, datasets
from eva.core.data.datamodules import schemas
from tests.eva.core.data.datamodules import _utils


def test_datamodule_methods(datamodule: datamodules.DataModule) -> None:
    """Tests the core datamodule methods."""

    def assert_dataset(
        dataset: datasets.TorchDataset | List[datasets.TorchDataset] | None, expected_called: bool
    ):
        datasets = dataset if isinstance(dataset, list) else [dataset]
        for ds in datasets:
            assert isinstance(ds, _utils.DummyDataset)
            assert ds._prepare_data_called is True
            assert ds._setup_called is expected_called
            assert ds._teardown_called is expected_called

    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.teardown(stage="fit")
    assert_dataset(datamodule.datasets.train, expected_called=True)
    assert_dataset(datamodule.datasets.val, expected_called=True)
    assert_dataset(datamodule.datasets.test, expected_called=False)
    assert_dataset(datamodule.datasets.predict, expected_called=False)

    datamodule.setup(stage="test")
    datamodule.teardown(stage="test")
    assert_dataset(datamodule.datasets.test, expected_called=True)

    datamodule.setup(stage="predict")
    datamodule.teardown(stage="predict")
    assert_dataset(datamodule.datasets.predict, expected_called=True)


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
            predict=[_utils.DummyDataset(), _utils.DummyDataset()],
        ),
        dataloaders=schemas.DataloadersSchema(
            train=dataloaders.DataLoader(),
            val=dataloaders.DataLoader(),
            test=dataloaders.DataLoader(),
            predict=dataloaders.DataLoader(),
        ),
    )
