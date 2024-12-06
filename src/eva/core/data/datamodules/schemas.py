"""Argument schemas used in DataModule."""

import dataclasses
from typing import List

from eva.core.data import dataloaders, datasets, samplers

TRAIN_DATASET = datasets.TorchDataset | None
"""Train dataset."""

EVAL_DATASET = datasets.TorchDataset | List[datasets.TorchDataset] | None
"""Evaluation dataset."""


@dataclasses.dataclass(frozen=True)
class DatasetsSchema:
    """Datasets schema used in DataModule."""

    train: TRAIN_DATASET = None
    """Train dataset."""

    val: EVAL_DATASET = None
    """Validation dataset."""

    test: EVAL_DATASET = None
    """Test dataset."""

    predict: EVAL_DATASET = None
    """Predict dataset."""

    def tolist(self, stage: str | None = None) -> List[EVAL_DATASET]:
        """Returns the dataclass as a list and optionally filters it given the stage."""
        match stage:
            case "fit":
                return [self.train, self.val]
            case "validate":
                return [self.val]
            case "test":
                return [self.test]
            case "predict":
                return [self.predict]
            case None:
                return [self.train, self.val, self.test, self.predict]
            case _:
                raise ValueError(f"Invalid stage `{stage}`.")


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


@dataclasses.dataclass(frozen=True)
class SamplersSchema:
    """Samplers schema used in DataModule."""

    train: samplers.Sampler | None = None
    """Train sampler."""

    val: samplers.Sampler | None = None
    """Validation sampler."""

    test: samplers.Sampler | None = None
    """Test sampler."""

    predict: samplers.Sampler | None = None
    """Predict sampler."""
