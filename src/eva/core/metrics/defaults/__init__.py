"""Default metric collections API."""

from eva.core.metrics.defaults.classification import (
    BinaryClassificationMetrics,
    MulticlassClassificationMetrics,
)
from eva.core.metrics.defaults.segmentation import MulticlassSegmentationMetrics

__all__ = [
    "MulticlassClassificationMetrics",
    "BinaryClassificationMetrics",
    "MulticlassSegmentationMetrics",
]
