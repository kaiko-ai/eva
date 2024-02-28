"""Core trainer module."""

from typing import Optional

from pytorch_lightning import trainer
from typing_extensions import override

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
    @override
    def log_dir(self) -> Optional[str]:
        """Overrides the log_dir getter from parent class."""
        if self._log_dir is not None:
            return self._log_dir
        else:
            return super().log_dir

    @log_dir.setter
    def log_dir(self, value):
        self._log_dir = value
