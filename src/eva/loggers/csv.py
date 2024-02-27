"""Overrides the CSVLogger class to add a log_dir setter."""

from pytorch_lightning import loggers
from typing_extensions import override


class CSVLogger(loggers.CSVLogger):
    """Overrides the CSVLogger class to add a log_dir setter."""

    def __init__(self, **kwargs):
        """Initializes a new CSVLogger instance."""
        super(CSVLogger, self).__init__(**kwargs)
        self._log_dir = None

    @property
    @override
    def log_dir(self) -> str:
        """Overrides the log_dir getter from parent class."""
        if self._log_dir is not None:
            return self._log_dir
        else:
            return super().log_dir

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value
