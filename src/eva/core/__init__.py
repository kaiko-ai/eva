"""eva core API."""

from eva.core.cli import cli
from eva.core.data import DataLoader, DataloadersSchema, DataModule, DatasetsSchema
from eva.core.interface import Interface
from eva.core.metrics import (
    AverageLoss,
    BinaryClassificationMetrics,
    MulticlassClassificationMetrics,
)
from eva.core.models import HeadModule, InferenceModule
from eva.core.trainers import Trainer

__all__ = [
    "AverageLoss",
    "BinaryClassificationMetrics",
    "MulticlassClassificationMetrics",
    "cli",
    "DataLoader",
    "DataloadersSchema",
    "DataModule",
    "DatasetsSchema",
    "Interface",
    "HeadModule",
    "InferenceModule",
    "Trainer",
]
