"""Default metric collections API."""

from eva.core.metrics.defaults.classification import (
    BinaryClassificationMetrics,
    MulticlassClassificationMetrics,
)

__all__ = [
    "MulticlassClassificationMetrics",
    "BinaryClassificationMetrics",
]
