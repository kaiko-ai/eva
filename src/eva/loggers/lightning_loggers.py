"""Custom logger classes for PyTorch Lightning."""

from pytorch_lightning import loggers
from typing_extensions import override


class BaseLogger(loggers.Logger):
    """Base logger class."""

    def __init__(self, *args, **kwargs):
        """Initializes the BaseLogger instance.

        Overwrites the parent class to allow for custom log_dir setting.
        """
        super().__init__(*args, **kwargs)
        self._log_dir = None

    @property
    @override
    def log_dir(self) -> str:
        if self._log_dir is not None:
            return self._log_dir
        else:
            return super().log_dir

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value


class TensorBoardLogger(BaseLogger, loggers.TensorBoardLogger):
    """TensorBoard logger class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CSVLogger(BaseLogger, loggers.CSVLogger):
    """CSV logger class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
