"""Core trainer module."""

from typing import Optional

from pytorch_lightning import trainer
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from eva.utils.recording import get_evaluation_id


class Trainer(trainer.Trainer):
    """Core Trainer class."""

    def __init__(self, **kwargs):
        """Initializes a new Trainer instance."""
        super(Trainer, self).__init__(**kwargs)
        self.evaluation_id = get_evaluation_id()
        self._log_dir = None
        self.i = 0

    @property
    def log_dir(self) -> Optional[str]:
        """Overrides the log_dir getter from parent class."""
        if self._log_dir is not None:
            return self._log_dir

        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], (TensorBoardLogger, CSVLogger)):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value
