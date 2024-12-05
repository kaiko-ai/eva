"""Base model module."""

import os
from typing import Any, Mapping

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import memory
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from eva.core.metrics import structs as metrics_lib
from eva.core.models.modules.typings import INPUT_BATCH
from eva.core.models.modules.utils import batch_postprocess


class ModelModule(pl.LightningModule):
    """The base model module."""

    def __init__(
        self,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocess: batch_postprocess.BatchPostProcess | None = None,
    ) -> None:
        """Initializes the basic module.

        Args:
            metrics: The metric groups to track.
            postprocess: A list of helper functions to apply after the
                loss and before the metrics calculation to the model
                predictions and targets.
        """
        super().__init__()

        self._metrics = metrics or self.default_metrics
        self._postprocess = postprocess or self.default_postprocess

        self.metrics = metrics_lib.MetricModule.from_schema(self._metrics)

    @property
    def default_metrics(self) -> metrics_lib.MetricsSchema:
        """The default metrics."""
        return metrics_lib.MetricsSchema()

    @property
    def default_postprocess(self) -> batch_postprocess.BatchPostProcess:
        """The default post-processes."""
        return batch_postprocess.BatchPostProcess()

    @property
    def metrics_device(self) -> torch.device:
        """Returns the device by which the metrics should be calculated."""
        device = os.getenv("METRICS_DEVICE", None)
        if device is not None:
            return torch.device(device)
        elif self.device.type == "mps":
            # mps seems to have compatibility issues with segmentation metrics
            return torch.device("cpu")
        return self.device

    @override
    def on_fit_start(self) -> None:
        self.metrics.to(device=self.metrics_device)

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
    def on_validation_start(self) -> None:
        self.metrics.to(device=self.metrics_device)

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
            outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics(self.metrics.validation_metrics)

    @override
    def on_test_start(self) -> None:
        self.metrics.to(device=self.metrics_device)

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
            outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics(self.metrics.test_metrics)

    def _common_batch_end(self, outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """Common end step of training, validation and test.

        It will apply the post-processes to the batch output and move
        them to the appropriate device.

        Args:
            outputs: The batch step outputs.

        Returns:
            The updated outputs.
        """
        self._postprocess(outputs)
        return memory.recursive_detach(outputs, to_cpu=self.metrics_device.type == "cpu")

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
        outputs: STEP_OUTPUT,
        dataloader_idx: int = 0,
    ) -> None:
        """Updates the metrics tracker with new data.

        Here the `outputs` keyword values will be filtered based
        on the signature of all individual metrics and passed only
        to the compatible ones.

        Args:
            metrics: The desired metrics tracker to update.
            outputs: The outputs of the batch processing step.
            dataloader_idx: The dataloader index.
        """
        inputs = self._parse_metrics_inputs(outputs, dataloader_idx)
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
        outputs: STEP_OUTPUT,
        dataloader_idx: int = 0,
    ) -> Mapping[str, Any]:
        """Parses the arguments for the metrics.

        When pass to a metrics collection object, the keyword values
        will be filtered based on the signature of all individual
        metrics and passed only to the compatible ones.

        Args:
            outputs: The outputs of the batch processing step.
            dataloader_idx: The dataloader index.

        Returns:
            A mapping with the argument name and its value.
        """
        if outputs is None:
            return {}

        if isinstance(outputs, torch.Tensor):
            outputs = {"loss": outputs}

        additional_metric_inputs = {
            "preds": outputs.get("predictions"),
            "target": outputs.get("targets"),
            "metadata": outputs.get("metadata"),
            "dataloader_idx": dataloader_idx,
        }
        return {**additional_metric_inputs, **outputs}
