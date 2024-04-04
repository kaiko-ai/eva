"""Core trainer module."""

import os
from typing import Any

import loguru
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import trainer as pl_trainer
from lightning.pytorch.utilities import argparse
from lightning_fabric.utilities import cloud_io
from typing_extensions import override

from eva.core import loggers as eva_loggers
from eva.core.data import datamodules
from eva.core.models import modules
from eva.core.trainers import _logging, functional


class Trainer(pl_trainer.Trainer):
    """Core trainer class.

    This is an extended version of lightning's core trainer class.
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

        For the input arguments, refer to ::class::`lightning.pytorch.Trainer`.

        Args:
            args: Positional arguments of ::class::`lightning.pytorch.Trainer`.
            default_root_dir: The default root directory to store the output logs.
                Unlike in ::class::`lightning.pytorch.Trainer`, this path would be the
                prioritized destination point.
            n_runs: The amount of runs (fit and evaluate) to perform in an evaluation session.
            kwargs: Kew-word arguments of ::class::`lightning.pytorch.Trainer`.
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

        enabled_loggers = []
        if isinstance(self.loggers, list) and len(self.loggers) > 0:
            for logger in self.loggers:
                if isinstance(logger, (pl_loggers.CSVLogger, pl_loggers.TensorBoardLogger)):
                    if not cloud_io._is_local_file_protocol(self.default_root_dir):
                        loguru.logger.warning(
                            f"Skipped {type(logger).__name__} as remote storage is not supported."
                        )
                        continue
                    else:
                        logger._root_dir = self.default_root_dir
                        logger._name = self._session_id
                        logger._version = subdirectory
                enabled_loggers.append(logger)

        self._loggers = enabled_loggers or [eva_loggers.DummyLogger(self._log_dir)]

    def run_evaluation_session(
        self,
        model: modules.ModelModule,
        datamodule: datamodules.DataModule,
    ) -> None:
        """Runs an evaluation session out-of-place.

        It performs an evaluation run (fit and evaluate) the model
        `self._n_run` times. Note that the input `base_model` would
        not be modified, so the weights of the input model will remain
        as they are.

        Args:
            model: The base model module to evaluate.
            datamodule: The data module.
        """
        functional.run_evaluation_session(
            base_trainer=self,
            base_model=model,
            datamodule=datamodule,
            n_runs=self._n_runs,
            verbose=self._n_runs > 1,
        )
