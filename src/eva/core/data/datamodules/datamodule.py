"""Core DataModule."""

from typing import List

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing_extensions import override

from eva.core.data import dataloaders as dataloaders_lib
from eva.core.data import datasets as datasets_lib
from eva.core.data.datamodules import call, schemas


class DataModule(pl.LightningDataModule):
    """DataModule encapsulates all the steps needed to process data.

    It will initialize and create the mapping between dataloaders and
    datasets. During the `prepare_data`, `setup` and `teardown`, the
    datamodule will call the respective methods from all datasets,
    given that they are defined.
    """

    def __init__(
        self,
        datasets: schemas.DatasetsSchema | None = None,
        dataloaders: schemas.DataloadersSchema | None = None,
    ) -> None:
        """Initializes the datamodule.

        Args:
            datasets: The desired datasets.
            dataloaders: The desired dataloaders.
        """
        super().__init__()

        self.datasets = datasets or self.default_datasets
        self.dataloaders = dataloaders or self.default_dataloaders

    @property
    def default_datasets(self) -> schemas.DatasetsSchema:
        """Returns the default datasets."""
        return schemas.DatasetsSchema()

    @property
    def default_dataloaders(self) -> schemas.DataloadersSchema:
        """Returns the default dataloader schema."""
        return schemas.DataloadersSchema()

    @override
    def prepare_data(self) -> None:
        call.call_method_if_exists(self.datasets.tolist(), "prepare_data")

    @override
    def setup(self, stage: str) -> None:
        call.call_method_if_exists(self.datasets.tolist(stage), "setup")

    @override
    def teardown(self, stage: str) -> None:
        call.call_method_if_exists(self.datasets.tolist(stage), "teardown")

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.datasets.train is None:
            raise ValueError(
                "Train dataloader can not be initialized as `self.datasets.train` is `None`."
            )
        return self.dataloaders.train(self.datasets.train)

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.datasets.val is None:
            raise ValueError(
                "Validation dataloader can not be initialized as `self.datasets.val` is `None`."
            )
        return self._initialize_dataloaders(self.dataloaders.val, self.datasets.val)

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.datasets.test is None:
            raise ValueError(
                "Test dataloader can not be initialized as `self.datasets.test` is `None`."
            )
        return self._initialize_dataloaders(self.dataloaders.test, self.datasets.test)

    @override
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.datasets.predict is None:
            raise ValueError(
                "Predict dataloader can not be initialized as `self.datasets.predict` is `None`."
            )
        return self._initialize_dataloaders(self.dataloaders.predict, self.datasets.predict)

    def _initialize_dataloaders(
        self,
        dataloader: dataloaders_lib.DataLoader,
        datasets: datasets_lib.TorchDataset | List[datasets_lib.TorchDataset],
    ) -> EVAL_DATALOADERS:
        """Initializes dataloaders from a given set of dataset.

        Args:
            dataloader: The dataloader to apply to the provided datasets.
            datasets: The desired dataset(s) to allocate dataloader(s).

        Returns:
            A list with the dataloaders of the provided dataset(s).
        """
        datasets = datasets if isinstance(datasets, list) else [datasets]
        return list(map(dataloader, datasets))
