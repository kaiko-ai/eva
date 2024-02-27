"""Default metric collections API."""

from eva.metrics.defaults.classification.binary import BinaryClassificationMetrics
from eva.metrics.defaults.classification.multiclass import MulticlassClassificationMetrics

__all__ = ["MulticlassClassificationMetrics", "BinaryClassificationMetrics"]
