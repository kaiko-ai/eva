"""Base batch callback logger."""

import abc

from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from eva.core.models.modules.typings import INPUT_TENSOR_BATCH


class BatchLogger(pl.Callback, abc.ABC):
    """Logs training and validation batch assets."""

    _batch_idx_to_log: int = 0
    """The batch index log."""

    def __init__(
        self,
        log_every_n_epochs: int | None = None,
        log_every_n_steps: int | None = None,
    ) -> None:
        """Initializes the callback object.

        Args:
            log_every_n_epochs: Epoch-wise logging frequency.
            log_every_n_steps: Step-wise logging frequency.
        """
        super().__init__()

        if log_every_n_epochs is None and log_every_n_steps is None:
            raise ValueError(
                "Please configure the logging frequency though "
                "`log_every_n_epochs` or `log_every_n_steps`."
            )
        if None not in [log_every_n_epochs, log_every_n_steps]:
            raise ValueError(
                "Arguments `log_every_n_epochs` and `log_every_n_steps` "
                "are mutually exclusive. Please configure one of them."
            )

        self._log_every_n_epochs = log_every_n_epochs
        self._log_every_n_steps = log_every_n_steps

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: INPUT_TENSOR_BATCH,
        batch_idx: int,
    ) -> None:
        if self._skip_logging(trainer, batch_idx if self._log_every_n_epochs else None):
            return

        self._log_batch(
            trainer=trainer,
            batch=batch,
            outputs=outputs,
            tag="BatchTrain",
        )

    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: INPUT_TENSOR_BATCH,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._skip_logging(trainer, batch_idx):
            return

        self._log_batch(
            trainer=trainer,
            batch=batch,
            outputs=outputs,
            tag="BatchValidation",
        )

    @abc.abstractmethod
    def _log_batch(
        self,
        trainer: pl.Trainer,
        outputs: STEP_OUTPUT,
        batch: INPUT_TENSOR_BATCH,
        tag: str,
    ) -> None:
        """Logs the batch data.

        Args:
            trainer: The trainer.
            outputs: The output of the train / val step.
            batch: The data batch.
            tag: The log tag.
        """

    def _skip_logging(
        self,
        trainer: pl.Trainer,
        batch_idx: int | None = None,
    ) -> bool:
        """Determines whether skip the logging step or not.

        Args:
            trainer: The trainer.
            batch_idx: The batch index.

        Returns:
            A boolean indicating whether to skip the step execution.
        """
        if trainer.global_step in [0, 1]:
            return False

        skip_due_frequency = any(
            [
                (trainer.current_epoch + 1) % (self._log_every_n_epochs or 1) != 0,
                (trainer.global_step + 1) % (self._log_every_n_steps or 1) != 0,
            ]
        )

        conditions = [
            skip_due_frequency,
            not trainer.is_global_zero,
            batch_idx != self._batch_idx_to_log if batch_idx else False,
        ]
        return any(conditions)
