"""Tests the embeddings writer."""

import itertools
import os
import random
import tempfile
from pathlib import Path
from typing import List, Literal

import pandas as pd
import pytest
import pytorch_lightning as pl
from pytorch_lightning.demos import boring_classes
from torch import nn
from typing_extensions import override

from eva.callbacks import writers
from eva.data import dataloaders, datamodules, datasets
from eva.models import modules


@pytest.mark.parametrize("batch_size, n_samples", list(itertools.product([5, 8], [7, 16])))
def test_embeddings_writer(datamodule: datamodules.DataModule, model: modules.HeadModule) -> None:
    """Tests the embeddings writer callback."""
    with tempfile.TemporaryDirectory() as output_dir:
        trainer = pl.Trainer(
            logger=False,
            callbacks=writers.EmbeddingsWriter(
                output_dir=output_dir, dataloader_idx_map={0: "train", 1: "val", 2: "test"}
            ),
        )
        all_predictions = trainer.predict(
            model=model, datamodule=datamodule, return_predictions=True
        )
        files = Path(output_dir).glob("*.pt")
        files = [f.relative_to(output_dir).as_posix() for f in files]

        assert isinstance(trainer.predict_dataloaders, list)
        assert len(trainer.predict_dataloaders) == 3
        assert isinstance(all_predictions, list)
        assert len(all_predictions) == 3
        total_n_predictions = 0
        for dataloader_idx in range(len(trainer.predict_dataloaders)):
            dataset = trainer.predict_dataloaders[dataloader_idx].dataset

            # Check if the number of predictions is correct
            predictions = all_predictions[dataloader_idx]
            assert isinstance(predictions, list)
            n_predictions = sum(len(p) for p in predictions)
            assert len(dataset) == n_predictions

            # Check if the expected files are present
            for idx in range(len(dataset)):
                filename = dataset.filename(idx)
                assert f"{filename}.pt" in files

            total_n_predictions += n_predictions

        # Check if the manifest file is in the expected format
        df_manifest = pd.read_csv(os.path.join(output_dir, "manifest.csv"))
        assert "filename" in df_manifest.columns
        assert "embedding" in df_manifest.columns
        assert "target" in df_manifest.columns
        assert "split" in df_manifest.columns
        assert len(df_manifest) == total_n_predictions


@pytest.fixture(scope="function")
def model(input_shape: int = 32, n_classes: int = 4) -> modules.HeadModule:
    """Returns a HeadModule model fixture."""
    return modules.HeadModule(
        head=nn.Linear(input_shape, n_classes),
        criterion=nn.CrossEntropyLoss(),
        backbone=nn.Flatten(),
    )


@pytest.fixture(scope="function")
def datamodule(
    dataset: List[datasets.Dataset],
    dataloader: dataloaders.DataLoader,
) -> datamodules.DataModule:
    """Returns a dummy classification datamodule fixture."""
    return datamodules.DataModule(
        datasets=datamodules.DatasetsSchema(
            predict=dataset,
        ),
        dataloaders=datamodules.DataloadersSchema(
            predict=dataloader,
        ),
    )


@pytest.fixture(scope="function")
def dataset(
    n_samples: int,
    sample_shape: int = 32,
) -> List[datasets.Dataset]:
    """Fake dataset fixture."""
    train_dataset = FakeDataset(split="train", length=n_samples, size=sample_shape)
    val_dataset = FakeDataset(split="val", length=n_samples, size=sample_shape)
    test_dataset = FakeDataset(split="test", length=n_samples, size=sample_shape)

    return [train_dataset, val_dataset, test_dataset]


@pytest.fixture(scope="function")
def dataloader(batch_size: int) -> dataloaders.DataLoader:
    """Test dataloader fixture."""
    return dataloaders.DataLoader(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )


class FakeDataset(boring_classes.RandomDataset, datasets.Dataset):
    """Fake prediction dataset."""

    def __init__(self, split: Literal["train", "val", "test"], size: int = 32, length: int = 10):
        """Initializes the dataset."""
        super().__init__(size=size, length=length)
        self._split = split

    def filename(self, index: int) -> str:
        """Returns the filename for the given index."""
        return f"{self._split}-{index}"

    @override
    def __getitem__(self, index: int):
        data = boring_classes.RandomDataset.__getitem__(self, index)
        target = random.choice([0, 1])
        return data, target
