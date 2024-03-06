"""Core trainer module."""

import os
from typing import Any

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import trainer as pl_trainer
from pytorch_lightning.utilities import argparse
from typing_extensions import override

from eva.trainers import utils


class Trainer(pl_trainer.Trainer):
    """Core trainer class.

    It is an extended version of lightning's core trainer class.
    """

    @argparse._defaults_from_env_vars
    def __init__(
        self,
        *args: Any,
        default_root_dir: str = "logs",
        **kwargs: Any,
    ) -> None:
        """Initializes the trainer.

        For the input arguments, refer to ::class::`pytorch_lightning.Trainer`.
        """
        super().__init__(*args, default_root_dir=default_root_dir, **kwargs)

        self._session_id: str = utils.generate_session_id()
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
