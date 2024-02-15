"""Tests the embeddings writer."""

import itertools
import os
import tempfile

import pandas as pd
import pytest
import pytorch_lightning as pl
from pytorch_lightning.demos import boring_classes

from eva.callbacks import writers
from eva.data import dataloaders, datamodules, datasets


@pytest.mark.parametrize("batch_size, n_samples", list(itertools.product([5, 8], [7, 16])))
def test_batch_prediction_writer(datamodule: datamodules.DataModule) -> None:
    """Tests the embeddings writer callback."""
    model = boring_classes.DemoModel(out_dim=10)

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = pl.Trainer(
            logger=False,
            callbacks=writers.BatchPredictionWriter(output_dir=temp_dir),
        )
        predictions = trainer.predict(model=model, datamodule=datamodule, return_predictions=True)
        files = os.listdir(temp_dir)

        assert isinstance(predictions, list)
        n_predictions = sum(len(p) for p in predictions)
        for file_idx in range(n_predictions):
            assert f"{file_idx}.pt" in files

        assert "manifest.csv" in files
        df_manifest = pd.read_csv(os.path.join(temp_dir, "manifest.csv"))

        assert "filename" in df_manifest.columns
        assert "prediction" in df_manifest.columns
        assert len(df_manifest) == n_predictions


@pytest.fixture(scope="function")
def datamodule(
    dataset: datasets.Dataset,
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
def dataloader(batch_size: int) -> dataloaders.DataLoader:
    """Test dataloader fixture."""
    return dataloaders.DataLoader(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )


@pytest.fixture(scope="function")
def dataset(
    n_samples: int,
    sample_shape: int = 32,
) -> datasets.Dataset:
    """Fake dataset fixture."""
    return FakeDataset(length=n_samples, size=sample_shape)


class FakeDataset(boring_classes.RandomDataset, datasets.Dataset):
    """Fake prediction dataset."""

    def filename(self, index: int) -> str:
        """Returns the filename for the given index."""
        return str(index)
