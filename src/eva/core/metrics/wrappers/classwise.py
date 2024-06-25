"""Wrapper metric to retrieve classwise metrics from metrics."""

from typing import Any

from torchmetrics import wrappers
from typing_extensions import override


class ClasswiseWrapper(wrappers.ClasswiseWrapper):
    """Wrapper metric for altering the output of classification metrics.

    It adds kwargs filtering during the update step for easy integration
    with `MetricCollection`.
    """

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        metric_kwargs = self.metric._filter_kwargs(**kwargs)
        return self._convert(self.metric(*args, **metric_kwargs))

    @override
    def update(self, *args: Any, **kwargs: Any) -> None:
        metric_kwargs = self.metric._filter_kwargs(**kwargs)
        self.metric.update(*args, **metric_kwargs)
