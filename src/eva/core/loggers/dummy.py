"""Dummy logger class."""

import lightning.pytorch.loggers.logger


class DummyLogger(lightning.pytorch.loggers.logger.DummyLogger):
    """Dummy logger class.

    This logger is currently used as a placeholder when saving results
    to remote storage, as common lightning loggers do not work
    with azure blob storage:

    <https://github.com/Lightning-AI/pytorch-lightning/issues/18861>
    <https://github.com/Lightning-AI/pytorch-lightning/issues/19736>

    Simply disabling the loggers when pointing to remote storage doesn't work
    because callbacks such as LearningRateMonitor or ModelCheckpoint require a
    logger to be present.
    """

    def __init__(self, save_dir: str) -> None:
        """Initializes the logger.

        Args:
            save_dir: The save directory (this logger does not save anything,
                but callbacks might use this path to save their outputs).
        """
        super().__init__()
        self._save_dir = save_dir

    @property
    def save_dir(self) -> str:
        """Returns the save directory."""
        return self._save_dir

    def __deepcopy__(self, memo=None):
        """Override of the deepcopy method."""
        return self
