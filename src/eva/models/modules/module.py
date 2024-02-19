"""Base model module."""

from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import memory
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

from eva.metrics import core as metrics_lib
from eva.models.modules.typings import INPUT_BATCH

from typing import Callable, List
import functools
import dataclasses
import torch

Transform = Callable[[torch.Tensor], torch.Tensor]

@dataclasses.dataclass(frozen=True)
class PostProcessesSchema:
    """Post-processes schema."""

    targets: List[Transform] | None = None
    """Holds the common train and evaluation metrics."""

    predictions: List[Transform] | None = None
    """Holds the common train and evaluation metrics."""

    def apply(self, tensor: torch.Tensor, transforms: List[Transform]) -> torch.Tensor:
        """Applies a list of transforms a specific group.

        Args:
            tensor: The desired tensor to process.
            transforms:

        Returns:
            The processed tensor.
        """
        return functools.reduce(
            lambda tensor, transform: transform(tensor), transforms, tensor
        )

    def __call__(self, outputs: STEP_OUTPUT) -> None:
        if not isinstance(outputs, dict):
            raise ValueError("")

        if "predictions" in outputs:
            outputs["predictions"] = self.apply(outputs["predictions"], self.predictions)

        if "targets" in outputs:
            outputs["targets"] = self.apply(outputs["targets"], self.targets)



class ModelModule(pl.LightningModule):
    """The base model module."""

    def __init__(
        self,
        metrics: metrics_lib.MetricsSchema | None = None,
        postprocesses: PostProcessesSchema | None = None,
    ) -> None:
        """Initializes the basic module.

        Args:
            metrics: The metrics schema.
        """
        super().__init__()

        self._metrics = metrics or self.default_metrics
        self._postprocesses = postprocesses

        self.metrics = metrics_lib.MetricModule.from_schema(self._metrics)
        self.postprocesses

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
            outputs=outputs,
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
            outputs=outputs,
            dataloader_idx=dataloader_idx,
        )

    @override
    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics(self.metrics.test_metrics)

    def _common_batch_end(self, outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """Common end step of training, validation and test.

        Args:
            outputs: The batch step outputs.

        Returns:
            The updated outputs.
        """
        return memory.recursive_detach(outputs, to_cpu=self.device.type == "cpu")

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
