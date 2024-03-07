"""Core trainer module."""

import os
from typing import Any

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import trainer as pl_trainer
from pytorch_lightning.utilities import argparse
from typing_extensions import override

from eva.data import datamodules
from eva.models import modules
from eva.trainers import _logging, functional


class Trainer(pl_trainer.Trainer):
    """Core trainer class.

    It is an extended version of lightning's core trainer class.
    """

    @argparse._defaults_from_env_vars
    def __init__(
        self,
        *args: Any,
        default_root_dir: str = "logs",
        n_runs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes the trainer.

        For the input arguments, refer to ::class::`pytorch_lightning.Trainer`.

        Args:
            args: Positional arguments of ::class::`pytorch_lightning.Trainer`.
            default_root_dir: The default root directory to store the output logs.
                In difference with ::class::`pytorch_lightning.Trainer`, this path
                would be the prior destination point.
            n_runs: The amount of runs (fit and evaluate) to perform in an evaluation session.
            kwargs: Kew-word arguments of ::class::`pytorch_lightning.Trainer`.
        """
        super().__init__(*args, default_root_dir=default_root_dir, **kwargs)

        self._n_runs = n_runs

        self._session_id: str = _logging.generate_session_id()
        self._log_dir: str = self.default_log_dir

        self.setup_log_dirs()

    @property
    def default_log_dir(self) -> str:
        """Returns the default log directory."""
        return os.path.join(self.default_root_dir, self._session_id)

    @property
    @override
    def log_dir(self) -> str | None:
        return self.strategy.broadcast(self._log_dir)

    def setup_log_dirs(self, subdirectory: str = "") -> None:
        """Setups the logging directory of the trainer and experimental loggers in-place.

        Args:
            subdirectory: Whether to append a subdirectory to the output log.
        """
        self._log_dir = os.path.join(self.default_root_dir, self._session_id, subdirectory)
        os.fspath(self._log_dir)

        for logger in self.loggers:
            if isinstance(logger, (pl_loggers.CSVLogger, pl_loggers.TensorBoardLogger)):
                logger._root_dir = self.default_root_dir
                logger._name = self._session_id
                logger._version = subdirectory

    def run_evaluation_session(
        self,
        model: modules.ModelModule,
        datamodule: datamodules.DataModule,
    ) -> None:
        """Runs a evaluation session out-of-place.

        It performs an evaluation run (fit and evaluate) the model
        `self._n_run` times. Note that the input `base_model` would
        not be modified, so the weights of the input model will remain
        as is.

        Args:
            model: The base model module to evaluate.
            datamodule: The data module.
        """
        functional.run_evaluation_session(
            base_trainer=self,
            base_model=model,
            datamodule=datamodule,
            n_runs=self._n_runs,
        )
