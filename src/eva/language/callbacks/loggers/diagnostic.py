"""Text prediction writer callbacks."""

import abc
import os
from typing import List, Sequence, TypedDict

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from typing_extensions import NotRequired, override

from eva.language.models.typings import TextBatch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from eva.multimodal.models.typings import TextImageBatch
from eva.core.loggers import log
from collections import deque
from loguru import logger

## THINGS NEEDED:
## DECODING OUTPUT - TAKE FROM CURRENT MODULE
## STORING RESULTS - DEQUEUE LIKE IN CURRENT MODULE
## LOGGING - TAKE FROM CURRENT MODULE, BASED ON INTERNAL STATES
## END GOAL: WORKING LOGGING + CLEAN MODULE


## FOLLOW-UP PR:
## IMPLEMENT METRIC BUFFER


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
            output_dir: The directory where the embeddings will be saved.
            model: The model instance used to generate the predictions.
            dataloader_idx_map: A dictionary mapping dataloader indices to their respective
                names (e.g. train, val, test).
            metadata_keys: An optional list of keys to extract from the batch metadata and store
                as additional columns in the manifest file.
            include_input: Whether to include the original input text messages in the output.
            overwrite: Whether to overwrite if embeddings are already present in the specified
                output directory. If set to `False`, an error will be raised if embeddings are
                already present (recommended).
            apply_postprocess: Whether to apply the postprocesses specified in the model module.
            save_format: The file format to use for saving the manifest file with the predictions.
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
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch_indices: Sequence[int],
        batch: TextBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        text_batch, target_batch, metadata_batch = self._unpack_batch(batch)

        decoded_input, decoded_output = self._decode_output(
            pl_module.model.processor, outputs["output_ids"], outputs["input_ids"].shape[-1]
        )

        for i in range(len(batch_indices)):
            entry: LoggingEntry = {
                "prompt": str(decoded_input[i]),
                "response": str(decoded_output[i]),
                "expected": str(target_batch[i]) if target_batch is not None else "",
                "objects": metadata_batch["objects"][i] if metadata_batch and "objects" in metadata_batch else [],
            }

            self._data.update(entry)

    def _decode_output(self, processor, output: torch.Tensor, instruction_length: int) -> List[str]:
        """Decode the model's batch output to text.

        Args:
            output: The raw output from the model.
            instruction_length: The length of the instruction in the input.

        Returns:
            A list of decoded text responses.
        """
        decoded_input = processor.batch_decode(  # type: ignore
            output[:, :instruction_length], skip_special_tokens=True
        )
        decoded_output = processor.batch_decode(  # type: ignore
            output[:, instruction_length:], skip_special_tokens=True
        )

        return decoded_input, decoded_output
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_generations(trainer, pl_module.metrics.validation_metrics)
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_generations(trainer, pl_module.metrics.test_metrics)
        return super().on_test_epoch_end(trainer, pl_module)

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
                    self._data[key] = attribute_value[-self.log_sample_size:]

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
            data=[list(item) for item in zip(*self._data.values())],
        )

    def _unpack_batch(self, batch: TextImageBatch | TextBatch):
        if isinstance(batch, TextImageBatch):
            return batch.text, batch.image, batch.target, batch.metadata
        return batch.text, None, batch.target, batch.metadata

