"""Metrics API."""

from eva.metrics.average_loss import AverageLoss
from eva.metrics.core import Metric, MetricCollection, MetricModule, MetricsSchema
from eva.metrics.defaults import MulticlassClassificationMetrics

__all__ = [
    "AverageLoss",
    "Metric",
    "MetricCollection",
    "MetricModule",
    "MetricsSchema",
    "MulticlassClassificationMetrics",
]
