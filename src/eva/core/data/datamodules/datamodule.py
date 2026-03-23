"""Core DataModule."""

from typing import List

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing_extensions import override

from eva.core.data import dataloaders as dataloaders_lib
from eva.core.data import datasets as datasets_lib
from eva.core.data import samplers as samplers_lib
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
        samplers: schemas.SamplersSchema | None = None,
    ) -> None:
        """Initializes the datamodule.

        Args:
            datasets: The desired datasets.
            dataloaders: The desired dataloaders.
            samplers: The desired samplers for the dataloaders.
        """
        super().__init__()

        self.datasets = datasets or self.default_datasets
        self.dataloaders = dataloaders or self.default_dataloaders
        self.samplers = samplers or self.default_samplers

    @property
    def default_datasets(self) -> schemas.DatasetsSchema:
        """Returns the default datasets."""
        return schemas.DatasetsSchema()

    @property
    def default_dataloaders(self) -> schemas.DataloadersSchema:
        """Returns the default dataloader schema."""
        return schemas.DataloadersSchema()

    @property
    def default_samplers(self) -> schemas.SamplersSchema:
        """Returns the default samplers schema."""
        return schemas.SamplersSchema()

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
        if isinstance(self.datasets.train, list) and len(self.datasets.train) > 1:
            raise ValueError("Train dataloader can not be initialized with multiple datasets.")

        return self._initialize_dataloaders(
            self.dataloaders.train, self.datasets.train, self.samplers.train
        )[0]

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.datasets.val is None:
            raise ValueError(
                "Validation dataloader can not be initialized as `self.datasets.val` is `None`."
            )
        return self._initialize_dataloaders(
            self.dataloaders.val, self.datasets.val, self.samplers.val
        )

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.datasets.test is None:
            raise ValueError(
                "Test dataloader can not be initialized as `self.datasets.test` is `None`."
            )
        return self._initialize_dataloaders(
            self.dataloaders.test, self.datasets.test, self.samplers.test
        )

    @override
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.datasets.predict is None:
            raise ValueError(
                "Predict dataloader can not be initialized as `self.datasets.predict` is `None`."
            )
        if isinstance(self.datasets.predict, list) and len(self.datasets.predict) > 1:
            # Only apply sampler to the first predict dataset (should correspond to train split)
            train_dataloader = self._initialize_dataloaders(
                self.dataloaders.predict, self.datasets.predict[0], self.samplers.predict
            )
            return train_dataloader + self._initialize_dataloaders(
                self.dataloaders.predict, self.datasets.predict[1:]
            )

        return self._initialize_dataloaders(
            self.dataloaders.predict, self.datasets.predict, self.samplers.predict
        )

    def _initialize_dataloaders(
        self,
        dataloader: dataloaders_lib.DataLoader,
        datasets: datasets_lib.TorchDataset | List[datasets_lib.TorchDataset],
        sampler: samplers_lib.Sampler | None = None,
    ) -> EVAL_DATALOADERS:
        """Initializes dataloaders from a given set of dataset.

        Args:
            dataloader: The dataloader to apply to the provided datasets.
            datasets: The desired dataset(s) to allocate dataloader(s).
            sampler: The sampler to use for the dataloader.

        Returns:
            A list with the dataloaders of the provided dataset(s).
        """
        datasets = datasets if isinstance(datasets, list) else [datasets]

        dataloaders = []
        for dataset in datasets:
            if sampler is not None and isinstance(sampler, samplers_lib.SamplerWithDataSource):
                sampler.set_dataset(dataset)  # type: ignore
            dataloaders.append(dataloader(dataset, sampler=sampler))
        return dataloaders
