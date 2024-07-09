"""Default classification metric collections API."""

from eva.core.metrics.defaults.classification.binary import BinaryClassificationMetrics
from eva.core.metrics.defaults.classification.multiclass import MulticlassClassificationMetrics

__all__ = ["BinaryClassificationMetrics", "MulticlassClassificationMetrics"]
