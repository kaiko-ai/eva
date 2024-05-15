"""Metrics related helper schemas."""

import dataclasses

from eva.core.metrics.structs.typings import MetricModuleType


@dataclasses.dataclass(frozen=True)
class MetricsSchema:
    """Metrics schema."""

    common: MetricModuleType | None = None
    """Holds the common train and evaluation metrics."""

    train: MetricModuleType | None = None
    """The exclusive training metrics."""

    evaluation: MetricModuleType | None = None
    """The exclusive evaluation metrics."""

    @property
    def training_metrics(self) -> MetricModuleType | None:
        """Returns the training metics."""
        return self._join_with_common(self.train)

    @property
    def evaluation_metrics(self) -> MetricModuleType | None:
        """Returns the evaluation metics."""
        return self._join_with_common(self.evaluation)

    def _join_with_common(self, metrics: MetricModuleType | None) -> MetricModuleType | None:
        """Joins the provided metrics with the common.

        Note that if there is duplication of metrics between the provided and the common
        (meaning there is the same metric in `metrics` and in `self.common`) both will
        be preserved.

        Args:
            metrics: The metrics to join.

        Returns:
            The resulted metrics after joining with the common ones.
        """
        if metrics is None or self.common is None:
            return self.common or metrics

        metrics = metrics if isinstance(metrics, list) else [metrics]  # type: ignore
        common = self.common if isinstance(self.common, list) else [self.common]
        return common + metrics  # type: ignore
