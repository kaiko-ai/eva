"""Base model module."""

from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import memory
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

from eva.metrics import core as metrics_lib
from eva.models.modules.typings import INPUT_BATCH


class ModelModule(pl.LightningModule):
    """The base model module."""

    def __init__(
        self,
        metrics: metrics_lib.MetricsSchema | None = None,
    ) -> None:
        """Initializes the basic module.

        Args:
            metrics: The metrics schema.
        """
        super().__init__()

        self._metrics = metrics or self.default_metrics

        self.metrics = metrics_lib.MetricModule.from_schema(self._metrics)

    @property
    def default_metrics(self) -> metrics_lib.MetricsSchema:
        """The default metrics."""
        return metrics_lib.MetricsSchema()

    @override
    def on_train_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: INPUT_BATCH,
        batch_idx: int,
    ) -> None:
        outputs = self._common_batch_end(outputs)
        self._forward_and_log_metrics(
            self.metrics.training_metrics,
            batch_outputs=outputs,
        )

    @override
    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: INPUT_BATCH,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        outputs = self._common_batch_end(outputs)
        self._update_metrics(
            self.metrics.validation_metrics,
            batch_outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics(self.metrics.validation_metrics)

    @override
    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: INPUT_BATCH,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        outputs = self._common_batch_end(outputs)
        self._update_metrics(
            self.metrics.test_metrics,
            batch_outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics(self.metrics.test_metrics)

    def _unpack_batch(self, batch: INPUT_BATCH) -> INPUT_BATCH:
        if isinstance(batch, (tuple, list)):
            data, targets = batch[:2]
            metadata = batch[2] if len(batch) == 3 else None
        else:
            raise ValueError(f"Invalid batch type: {type(batch)}")

        return data, targets, metadata  # type: ignore

    def _common_batch_end(self, outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """Common end step of training, validation and test.

        Args:
            outputs: The batch step outputs.

        Returns:
            The updated outputs.
        """
        return memory.recursive_detach(
            outputs,
            to_cpu=self.device.type == "cpu",
        )

    def _forward_and_log_metrics(
        self,
        metrics: metrics_lib.MetricCollection,
        batch_outputs: STEP_OUTPUT,
    ) -> None:
        """Performs a forward pass to the metrics and logs them.

        Args:
            metrics: The desired metrics tracker to log.
            batch_outputs: The outputs of the batch processing step.
        """
        inputs = self._parse_metrics_inputs(batch_outputs)
        metrics(**inputs)
        self.log_dict(metrics, on_step=True, on_epoch=False)

    def _update_metrics(
        self,
        metrics: metrics_lib.MetricCollection,
        batch_outputs: STEP_OUTPUT,
        dataloader_idx: int = 0,
    ) -> None:
        """Updates the metrics tracker with new data.

        Here the `batch_outputs` keyword values will be filtered based
        on the signature of all individual metrics and passed only
        to the compatible ones.

        Args:
            metrics: The desired metrics tracker to update.
            batch_outputs: The outputs of the batch processing step.
            dataloader_idx: The dataloader index.
        """
        inputs = self._parse_metrics_inputs(batch_outputs, dataloader_idx)
        metrics.update(**inputs)

    def _compute_and_log_metrics(self, metrics: metrics_lib.MetricCollection) -> None:
        """Computes, logs and resets the metrics.

        Args:
            metrics: The desired metrics tracker to log.
        """
        outputs = metrics.compute()
        self.log_dict(outputs)
        metrics.reset()

    def _parse_metrics_inputs(
        self,
        batch_outputs: STEP_OUTPUT,
        dataloader_idx: int = 0,
    ) -> Mapping[str, Any]:
        """Parses the arguments for the metrics.

        When pass to a metrics collection object, the keyword values
        will be filtered based on the signature of all individual
        metrics and passed only to the compatible ones.

        Args:
            batch_outputs: The outputs of the batch processing step.
            dataloader_idx: The dataloader index.

        Returns:
            A mapping with the argument name and its value.
        """
        if batch_outputs is None:
            return {}

        if isinstance(batch_outputs, torch.Tensor):
            batch_outputs = {"loss": batch_outputs}

        additional_metric_inputs = {
            "preds": batch_outputs.get("predictions"),
            "target": batch_outputs.get("targets"),
            "metadata": batch_outputs.get("metadata"),
            "dataloader_idx": dataloader_idx,
        }
        return {**additional_metric_inputs, **batch_outputs}
