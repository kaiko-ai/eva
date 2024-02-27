"""Overrides the TensorBoardLogger class to add a log_dir setter."""

import os

from pytorch_lightning import loggers


class TensorBoardLogger(loggers.TensorBoardLogger):
    """Overrides the TensorBoardLogger class to add a log_dir setter."""

    def __init__(self, **kwargs):
        """Initializes a new TensorBoardLogger instance."""
        super(TensorBoardLogger, self).__init__(**kwargs)
        self._log_dir = None

    @property
    def log_dir(self) -> str:
        """Overrides the log_dir getter from parent class."""
        if self._log_dir is not None:
            return self._log_dir

        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        self._log_dir = os.path.expanduser(log_dir)
        return self._log_dir

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value
