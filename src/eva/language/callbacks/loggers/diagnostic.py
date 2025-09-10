"""Text prediction writer callbacks."""

import abc
import os
from collections import deque
from typing import List, TypedDict

import lightning.pytorch as pl
from lightning.pytorch import callbacks
from loguru import logger
from typing_extensions import NotRequired, override

from eva.core.loggers import log
from eva.core.metrics import structs as metrics_lib
from eva.language.models.typings import TextBatch
from eva.multimodal.models.typings import TextImageBatch


class LoggingEntry(TypedDict):
    """A single entry in the logging file."""

    prompt: str
    """The input prompt text."""

    response: str
    """The generated response text."""

    expected: str
    """The expected response text."""

    objects: NotRequired[List[str]]
    """A list of objects detected in the input."""


class DiagnosticLoggerCallback(callbacks.Callback, abc.ABC):
    """Callback for logging diagnostic information during training and evaluation."""

    def __init__(
        self,
        log_generations: bool = True,
        log_sample_size: int = 100,
    ) -> None:
        """Initializes a new callback.

        Args:
            log_generations: Whether to log the generated text & samplewise metrics.
            log_sample_size: The number of samples to log if `log_generations` is True
        """
        super().__init__()

        self.log_generations = log_generations
        self.log_sample_size = log_sample_size
        self.log_counter = 0

        self.log_sample_size = log_sample_size
        if self.log_generations:
            self._data = {
                "prompt": deque(maxlen=log_sample_size),
                "response": deque(maxlen=log_sample_size),
                "expected": deque(maxlen=log_sample_size),
                "objects": deque(maxlen=log_sample_size),
            }
            self.task_name = os.getenv("TASK_NAME", "multimodal")

    def _batch_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pass

    @override
    def on_validation_end(self, trainer, pl_module):
        self._log_generations(trainer, pl_module.metrics.validation_metrics)
        return super().on_validation_end(trainer, pl_module)

    @override
    def on_test_end(self, trainer, pl_module):
        self._log_generations(trainer, pl_module.metrics.test_metrics)
        return super().on_test_end(trainer, pl_module)

    def _log_generations(self, trainer, metrics: metrics_lib.MetricCollection | None):
        """Logs the generated text & samplewise metrics."""
        # Log scores from metrics
        if metrics is not None:
            for metric in metrics.children():
                metric_name = repr(metric)
                attribute_names = metric.output.keys()

                for attribute_name in attribute_names:
                    attribute_value = metric.output[attribute_name]
                    key = (
                        f"{metric_name}_{attribute_name}"
                        if attribute_name not in self._data
                        else attribute_name
                    )
                    self._data[key] = attribute_value[-self.log_sample_size :]

        # Remove keys with value length 0 and log a warning
        keys_to_remove = [k for k, v in self._data.items() if len(v) == 0]
        for k in keys_to_remove:
            del self._data[k]
            logger.warning(f"Key '{k}' has zero length and was removed from logging.")

        lengths = [len(v) for v in self._data.values()]
        if not lengths or len(set(lengths)) != 1:
            raise ValueError(
                "Not all logged items have the same length. Skipping logging generations."
            )

        log.log_table(
            trainer.loggers,
            tag=f"test/{self.task_name} scored generations",
            columns=list(self._data.keys()),
            data=[list(item) for item in zip(*self._data.values(), strict=False)],
        )

    def _unpack_batch(self, batch: TextImageBatch | TextBatch):
        if isinstance(batch, TextImageBatch):
            return batch.text, batch.image, batch.target, batch.metadata
        return batch.text, None, batch.target, batch.metadata
