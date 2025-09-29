"""Tests the text prediction writer."""

import functools
import os
import tempfile
from typing import List, Literal, cast

import lightning.pytorch as pl
import pandas as pd
import pytest
from lightning.pytorch import Callback
from lightning.pytorch.demos import boring_classes
from torch import nn
from typing_extensions import override

from eva.core.data import dataloaders, datamodules, datasets
from eva.language.callbacks.writers import prediction as prediction_writer
from eva.language.data.dataloaders import text_collate
from eva.language.data.datasets.typings import TextSample
from eva.language.data.messages import UserMessage
from eva.language.models.typings import ModelOutput


@pytest.mark.parametrize(
    "batch_size, n_samples, save_format, include_input",
    [
        (5, 7, "jsonl", True),
        (8, 16, "csv", True),
        (8, 32, "parquet", True),
        (5, 7, "jsonl", False),
    ],
)
def test_prediction_writer(
    datamodule: datamodules.DataModule,
    model: pl.LightningModule,
    text_generation_model: nn.Module,
    save_format: Literal["jsonl", "parquet", "csv"],
    include_input: bool,
) -> None:
    """Tests the text prediction writer callback.

    This test executes a lightning trainer predict operation and checks if the expected
    predictions are correctly written to disk.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        callback = prediction_writer.TextPredictionWriter(
            output_dir=output_dir,
            model=text_generation_model,
            dataloader_idx_map={0: "train", 1: "val", 2: "test"},
            metadata_keys=["example_metadata"],
            include_input=include_input,
            overwrite=True,
            save_format=save_format,
        )
        trainer = _init_and_run_trainer([callback], model, datamodule)

        assert isinstance(trainer.predict_dataloaders, list)
        assert len(trainer.predict_dataloaders) == 3

        _check_manifest(output_dir, datamodule, save_format, include_input)


@pytest.mark.parametrize("batch_size, n_samples, save_format", [(5, 7, "jsonl")])
def test_prediction_writer_overwrite_protection(
    datamodule: datamodules.DataModule,
    model: pl.LightningModule,
    text_generation_model: nn.Module,
    save_format: Literal["jsonl", "parquet", "csv"],
) -> None:
    """Tests that the writer raises an error when overwrite is False and files exist."""
    with tempfile.TemporaryDirectory() as output_dir:
        callback = prediction_writer.TextPredictionWriter(
            output_dir=output_dir,
            model=text_generation_model,
            overwrite=True,
            save_format=save_format,
        )
        _init_and_run_trainer([callback], model, datamodule)

        # Try to write again without overwrite
        callback2 = prediction_writer.TextPredictionWriter(
            output_dir=output_dir,
            model=text_generation_model,
            overwrite=False,
            save_format=save_format,
        )

        with pytest.raises(FileExistsError):
            _init_and_run_trainer([callback2], model, datamodule)


def _init_and_run_trainer(
    callbacks: List[Callback],
    model: pl.LightningModule,
    datamodule: datamodules.DataModule,
):
    """Initializes and runs the trainer with the given callbacks."""
    trainer = pl.Trainer(
        logger=False,
        accelerator="cpu",
        callbacks=callbacks,
    )
    trainer.predict(model=model, datamodule=datamodule, return_predictions=True)

    return trainer


def _check_manifest(
    output_dir: str,
    datamodule: datamodules.DataModule,
    save_format: Literal["jsonl", "parquet", "csv"],
    include_input: bool = True,
):
    """Checks if the manifest file contains the expected entries."""
    manifest_path = os.path.join(output_dir, f"manifest.{save_format}")
    assert os.path.isfile(manifest_path)

    # Load manifest based on format
    match save_format:
        case "jsonl":
            df_manifest = pd.read_json(manifest_path, lines=True)
        case "csv":
            df_manifest = pd.read_csv(manifest_path)
        case "parquet":
            df_manifest = pd.read_parquet(manifest_path)
        case _:
            raise ValueError(f"Unsupported save format: {save_format}")

    # Check expected columns
    expected_columns = ["prediction", "target", "split", "example_metadata"]
    for column in expected_columns:
        assert column in df_manifest.columns

    if include_input:
        assert "text" in df_manifest.columns

    # Check number of entries
    total_samples = sum(len(ds) for ds in datamodule.datasets.predict)  # type: ignore
    assert len(df_manifest) == total_samples

    # Check that predictions are strings
    assert all(isinstance(pred, str) for pred in df_manifest["prediction"])

    # Check that splits are correctly assigned
    assert set(df_manifest["split"]) == {"train", "val", "test"}


@pytest.fixture(scope="function")
def text_generation_model() -> nn.Module:
    """Returns a simple model that generates text predictions."""
    return FakeTextModel()


@pytest.fixture(scope="function")
def model(text_generation_model: nn.Module) -> pl.LightningModule:
    """Returns a LightningModule wrapper for the text model."""
    return FakeLightningModule(text_generation_model)


@pytest.fixture(scope="function")
def dataset(n_samples: int) -> List[datasets.TorchDataset]:
    """Fake dataset fixture."""
    Dataset = functools.partial(
        FakeTextDataset,
        length=n_samples,
    )
    train_dataset = Dataset(split="train")
    val_dataset = Dataset(split="val")
    test_dataset = Dataset(split="test")

    return [train_dataset, val_dataset, test_dataset]


@pytest.fixture(scope="function")
def datamodule(batch_size: int, dataset: List[datasets.TorchDataset]) -> datamodules.DataModule:
    """Returns a DataModule fixture."""
    dataloader = dataloaders.DataLoader(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        collate_fn=text_collate,
    )
    return datamodules.DataModule(
        datasets=datamodules.DatasetsSchema(
            train=dataset[0], val=dataset[1], predict=cast(List[datasets.TorchDataset], dataset)
        ),
        dataloaders=datamodules.DataloadersSchema(
            train=dataloader,
            val=dataloader,
            predict=dataloader,
        ),
    )


class FakeTextModel(nn.Module):
    """Fake model that generates text predictions."""

    def forward(self, batch):
        """Returns a list of fake predictions."""
        text_batch, _, _ = batch
        batch_size = len(text_batch)
        predictions = [f"prediction_{i}" for i in range(batch_size)]
        return ModelOutput(generated_text=predictions)


class FakeLightningModule(pl.LightningModule):
    """Fake LightningModule wrapper."""

    def __init__(self, text_model: nn.Module):
        """Initializes the module."""
        super().__init__()
        self.text_model = text_model

    def forward(self, x) -> ModelOutput:
        """Forward pass."""
        return ModelOutput(generated_text=self.text_model(x))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step."""
        return self(batch)


class FakeTextDataset(boring_classes.RandomDataset, datasets.Dataset):
    """Fake text dataset."""

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        length: int = 10,
    ):
        """Initializes the dataset."""
        super().__init__(size=32, length=length)
        self._split = split

    @override
    def __getitem__(self, index: int):
        """Returns a text sample with metadata."""
        text = [UserMessage(content=f"Sample text {self._split}-{index}")]
        target = f"target_{index}"
        metadata = {"example_metadata": f"metadata_{index}"}
        return TextSample(text=text, target=target, metadata=metadata)  # type: ignore
