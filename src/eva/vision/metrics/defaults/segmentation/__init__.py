"""Default segmentation metric collections API."""

from eva.vision.metrics.defaults.segmentation.multiclass import (
    MulticlassSegmentationMetrics,
    MulticlassSegmentationMetricsV2,
)

__all__ = ["MulticlassSegmentationMetrics", "MulticlassSegmentationMetricsV2"]
