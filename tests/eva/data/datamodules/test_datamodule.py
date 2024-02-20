"""Tests for the core datamodule."""

from typing import List

import pytest
from torch.utils import data as torch_data

from eva.data import dataloaders, datamodules, datasets
from eva.data.datamodules import schemas
from tests.eva.data.datamodules import _utils


def test_datamodule_methods_fit(datamodule: datamodules.DataModule) -> None:
    """Tests if fit stage correctly initializes the train & val datasets."""
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.teardown(stage="fit")

    _assert_dataset(datamodule.datasets.train)
    _assert_dataset(datamodule.datasets.val)


def test_datamodule_methods_validate(datamodule: datamodules.DataModule) -> None:
    """Tests if fit stage correctly initializes the val dataset."""
    datamodule.prepare_data()
    datamodule.setup(stage="validate")
    datamodule.teardown(stage="validate")

    _assert_dataset(datamodule.datasets.val)


def test_datamodule_methods_test(datamodule: datamodules.DataModule) -> None:
    """Tests if fit stage correctly initializes the test dataset."""
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    datamodule.teardown(stage="test")

    _assert_dataset(datamodule.datasets.test)


def test_datamodule_methods_predict(datamodule: datamodules.DataModule) -> None:
    """Tests if fit stage correctly initializes the predict dataset."""
    datamodule.prepare_data()
    datamodule.setup(stage="predict")
    datamodule.teardown(stage="predict")

    _assert_dataset(datamodule.datasets.predict)


def test_datamodule_dataloaders(datamodule: datamodules.DataModule) -> None:
    """Tests the core datamodule dataloaders."""

    def _assert_evaluation_dataloader(dataloaders: List[torch_data.DataLoader]) -> None:
        """Asserts the evaluation dataloaders."""
        assert isinstance(dataloaders, list) and isinstance(dataloaders[0], torch_data.DataLoader)

    assert isinstance(datamodule.train_dataloader(), torch_data.DataLoader)
    _assert_evaluation_dataloader(datamodule.val_dataloader())
    _assert_evaluation_dataloader(datamodule.test_dataloader())
    _assert_evaluation_dataloader(datamodule.predict_dataloader())


def _assert_dataset(dataset: datasets.Dataset | List[datasets.Dataset] | None):
    datasets = dataset if isinstance(dataset, list) else [dataset]
    for ds in datasets:
        assert isinstance(ds, _utils.DummyDataset)
        assert ds._prepare_data_called is True
        assert ds._setup_called is True
        assert ds._teardown_called is True


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
