"""Metric typings."""

from typing import Dict, Sequence, Union

from eva.metrics.core import collection, metric

BaseMetricModuleType = Union[metric.Metric, collection.MetricCollection]
"""The base module metric type."""

MetricModuleType = Union[
    BaseMetricModuleType,
    Sequence[BaseMetricModuleType],
    Dict[str, BaseMetricModuleType],
]
"""The module metric type."""
