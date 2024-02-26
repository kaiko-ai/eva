import os

from pytorch_lightning import loggers


class CSVLogger(loggers.CSVLogger):
    """Overrides the CSVLogger class to add a log_dir setter."""

    def __init__(self, **kwargs):
        super(CSVLogger, self).__init__(**kwargs)
        self._log_dir = None

    @property
    def log_dir(self) -> str:
        """Overrides the log_dir getter from parent class."""
        if self._log_dir is not None:
            return self._log_dir

        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value
