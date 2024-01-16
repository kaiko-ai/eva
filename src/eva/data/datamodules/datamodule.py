"""DataModule."""
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing_extensions import override

from eva.data import dataloaders as dataloaders_lib
from eva.data import datasets as datasets_lib
from eva.data.datamodules import call, schemas


class DataModule(pl.LightningDataModule):
    """DataModule."""

    def __init__(
        self,
        datasets: schemas.DatasetsSchema | None = None,
        dataloaders: schemas.DataloadersSchema | None = None,
    ) -> None:
        """Initializes the datamodule.

        Args:
            datasets: The desired datasets. Defaults to `None`.
            dataloaders: The desired dataloaders. Defaults to `None`.
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
        call.call_method_if_exists(self.datasets, "prepare_data")

    @override
    def setup(self, stage: str) -> None:
        call.call_method_if_exists(self.datasets, "setup")

    @override
    def teardown(self, stage: str) -> None:
        call.call_method_if_exists(self.datasets, "teardown")

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
        datasets: datasets_lib.Dataset | List[datasets_lib.Dataset],
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
