"""Metrics module."""

from __future__ import annotations

from torch import nn

from eva.core.metrics.structs import collection, schemas
from eva.core.metrics.structs.typings import MetricModuleType


class MetricModule(nn.Module):
    """The metrics module.

    Allows to store and keep track of `train`, `val` and `test` metrics.
    """

    def __init__(
        self,
        train: collection.MetricCollection | None,
        val: collection.MetricCollection | None,
        test: collection.MetricCollection | None,
    ) -> None:
        """Initializes the metrics for the Trainer.

        Args:
            train: The training metric collection.
            val: The validation metric collection.
            test: The test metric collection.
        """
        super().__init__()

        self._train = train or self.default_metric_collection
        self._val = val or self.default_metric_collection
        self._test = test or self.default_metric_collection

    @property
    def default_metric_collection(self) -> collection.MetricCollection:
        """Returns the default metric collection."""
        return collection.MetricCollection([])

    @classmethod
    def from_metrics(
        cls,
        train: MetricModuleType | None,
        val: MetricModuleType | None,
        test: MetricModuleType | None,
        *,
        separator: str = "/",
    ) -> MetricModule:
        """Initializes a metric module from a list of metrics.

        Args:
            train: Metrics for the training stage.
            val: Metrics for the validation stage.
            test: Metrics for the test stage.
            separator: The separator between the group name of the metric
                and the metric itself.
        """
        return cls(
            train=_create_collection_from_metrics(train, prefix="train" + separator),
            val=_create_collection_from_metrics(val, prefix="val" + separator),
            test=_create_collection_from_metrics(test, prefix="test" + separator),
        )

    @classmethod
    def from_schema(
        cls,
        schema: schemas.MetricsSchema,
        *,
        separator: str = "/",
    ) -> MetricModule:
        """Initializes a metric module from the metrics schema.

        Args:
            schema: The dataclass metric schema.
            separator: The separator between the group name of the metric
                and the metric itself.
        """
        return cls.from_metrics(
            train=schema.training_metrics,
            val=schema.evaluation_metrics,
            test=schema.evaluation_metrics,
            separator=separator,
        )

    @property
    def training_metrics(self) -> collection.MetricCollection:
        """Returns the metrics of the train dataset."""
        return self._train

    @property
    def validation_metrics(self) -> collection.MetricCollection:
        """Returns the metrics of the validation dataset."""
        return self._val

    @property
    def test_metrics(self) -> collection.MetricCollection:
        """Returns the metrics of the test dataset."""
        return self._test


def _create_collection_from_metrics(
    metrics: MetricModuleType | None, *, prefix: str | None = None
) -> collection.MetricCollection | None:
    """Create a unique collection from metrics.

    Args:
        metrics: The desired metrics.
        prefix: A prefix to added to the collection.

    Returns:
        The resulted metrics collection.
    """
    metrics_collection = collection.MetricCollection(metrics or [], prefix=prefix)  # type: ignore
    return metrics_collection.clone()
