"""Experimental loggers."""

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

Loggers = TensorBoardLogger | WandbLogger
"""Supported loggers."""
