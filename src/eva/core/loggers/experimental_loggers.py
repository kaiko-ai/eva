"""Experiment loggers."""

from typing import Union

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

"""Supported loggers."""
ExperimentalLoggers = Union[CSVLogger, TensorBoardLogger, WandbLogger]
