"""Argument schemas used in DataModule."""
import dataclasses
from typing import List

from eva.data import dataloaders, datasets


@dataclasses.dataclass(frozen=True)
class DatasetsSchema:
    """Datasets schema used in DataModule."""

    train: datasets.Dataset | None = None
    """Train dataset."""

    val: datasets.Dataset | List[datasets.Dataset] | None = None
    """Validation dataset."""

    test: datasets.Dataset | List[datasets.Dataset] | None = None
    """Test dataset."""

    predict: datasets.Dataset | List[datasets.Dataset] | None = None
    """Predict dataset."""


@dataclasses.dataclass(frozen=True)
class DataloadersSchema:
    """Dataloaders schema used in DataModule."""

    train: dataloaders.DataLoader = dataclasses.field(default_factory=dataloaders.DataLoader)
    """Train dataloader."""

    val: dataloaders.DataLoader = dataclasses.field(default_factory=dataloaders.DataLoader)
    """Validation dataloader."""

    test: dataloaders.DataLoader = dataclasses.field(default_factory=dataloaders.DataLoader)
    """Test dataloader."""

    predict: dataloaders.DataLoader = dataclasses.field(default_factory=dataloaders.DataLoader)
    """Predict dataloader."""
