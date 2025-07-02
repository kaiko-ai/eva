"""Default metric collections API."""

from eva.vision.metrics.defaults.segmentation import (
    MulticlassSegmentationMetrics,
    MulticlassSegmentationMetricsV2,
)

__all__ = [
    "MulticlassSegmentationMetrics",
    "MulticlassSegmentationMetricsV2",
]
