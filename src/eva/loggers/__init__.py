"""Loggers API."""

from eva.loggers.csv import CSVLogger
from eva.loggers.tensorboard import TensorBoardLogger

__all__ = ["CSVLogger", "TensorBoardLogger"]
