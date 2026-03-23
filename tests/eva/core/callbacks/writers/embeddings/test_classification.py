"""Tests the embeddings writer."""

import functools
import os
import random
import tempfile
from pathlib import Path
from typing import List, Literal, Set

import lightning.pytorch as pl
import pandas as pd
import pytest
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.demos import boring_classes
from torch import nn
from typing_extensions import override

from eva.core.callbacks import writers
from eva.core.data import datamodules, datasets
from eva.core.models import modules

SAMPLE_SHAPE = 32


@pytest.mark.parametrize(
    "batch_size, n_samples, metadata_keys, filenames",
    [
        (5, 7, None, None),
        (5, 7, ["wsi_id"], None),
        (8, 16, None, None),
        (8, 32, ["wsi_id"], ["slide_1", "slide_2"]),
    ],
)
def test_embeddings_writer(datamodule: datamodules.DataModule, model: modules.HeadModule) -> None:
    """Tests the embeddings writer callback.

    This test executes a lightning trainer predict operation and checks if the expected
    embedding tensors & manifest files are correctly written to disk.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        metadata_keys = datamodule.datasets.predict[0]._metadata_keys  # type: ignore
        expected_filenames = datamodule.datasets.predict[0]._filenames  # type: ignore
        grouping_enabled = expected_filenames is not None
        callback = writers.ClassificationEmbeddingsWriter(
            output_dir=output_dir,
            dataloader_idx_map={0: "train", 1: "val", 2: "test"},
            backbone=nn.Flatten(),
            metadata_keys=metadata_keys,
            overwrite=True,
        )
        trainer = _init_and_run_trainer([callback], model, datamodule)

        assert isinstance(trainer.predict_dataloaders, list)
        assert len(trainer.predict_dataloaders) == 3

        unique_filenames = set()
        tot_n_samples = 0
        for dataloader_idx in range(len(trainer.predict_dataloaders)):
            _check_embedding_dimensions(output_dir, grouping_enabled)
            dataset = trainer.predict_dataloaders[dataloader_idx].dataset
            filenames = _check_if_embedding_files_exist(output_dir, dataset, expected_filenames)
            unique_filenames.update(filenames)
            tot_n_samples += len(dataset)

        expected_file_count = len(unique_filenames) if expected_filenames else tot_n_samples
        _check_expected_n_files(output_dir, expected_file_count)
        _check_manifest(output_dir, len(unique_filenames), metadata_keys)


def _init_and_run_trainer(
    callbacks: List[callbacks.Callback],
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


def _check_if_embedding_files_exist(
    output_dir: str, dataset: datasets.Dataset, expected_filenames: List[str] | None
) -> Set[str]:
    """Checks if the expected embedding files exist in the output directory."""
    output_files = _get_output_filenames(output_dir)

    dataset_filenames = set()
    for idx in range(len(dataset)):  # type: ignore
        filename = f"{dataset.filename(idx)}.pt"  # type: ignore
        assert filename in output_files
        dataset_filenames.add(filename)

    if expected_filenames:
        assert len(set(expected_filenames) - {Path(x).stem for x in output_files}) == 0

    return dataset_filenames


def _check_embedding_dimensions(output_dir: str, grouping_enabled: bool):
    """Checks if the produced embeddings have the expected dimensions."""
    embedding_paths = Path(output_dir).glob("*.pt")

    for path in embedding_paths:
        tensor_list = torch.load(path)
        assert isinstance(tensor_list, list)
        for t in tensor_list:
            assert isinstance(t, torch.Tensor)
            assert t.ndim == 1

        if grouping_enabled:
            assert len(tensor_list) > 1
        else:
            assert len(tensor_list) == 1


def _check_expected_n_files(output_dir: str, expected_file_count: int):
    """Checks if the number of produced output files matches the expected count."""
    output_files = _get_output_filenames(output_dir)
    assert len(output_files) == expected_file_count


def _check_manifest(
    output_dir: str, expected_n_entries: int, metadata_keys: List[str] | None = None
):
    """Checks if the manifest file contains the expected number of entries and columns."""
    manifest_path = os.path.join(output_dir, "manifest.csv")
    assert os.path.isfile(manifest_path)
    df_manifest = pd.read_csv(manifest_path)

    expected_columns = ["origin", "embeddings", "target", "split"] + (metadata_keys or [])
    for column in expected_columns:
        assert column in df_manifest.columns

    assert len(df_manifest) == expected_n_entries

    if metadata_keys:
        assert all(key in df_manifest.columns for key in metadata_keys)


def _get_output_filenames(output_dir: str) -> List[str]:
    """Returns the list of output embedding filenames in the output directory."""
    output_files = Path(output_dir).glob("*.pt")
    output_files = [f.relative_to(output_dir).as_posix() for f in output_files]
    return output_files


@pytest.fixture(scope="function")
def model(n_classes: int = 4) -> modules.HeadModule:
    """Returns a HeadModule model fixture."""
    return modules.HeadModule(
        head=nn.Linear(SAMPLE_SHAPE, n_classes),
        criterion=nn.CrossEntropyLoss(),
        backbone=None,
    )


@pytest.fixture(scope="function")
def dataset(
    n_samples: int,
    metadata_keys: List[str] | None,
    filenames: List[str] | None,
) -> List[datasets.Dataset]:
    """Fake dataset fixture."""
    Dataset = functools.partial(
        FakeDataset,
        length=n_samples,
        size=SAMPLE_SHAPE,
        metadata_keys=metadata_keys,
        filenames=filenames,
    )
    train_dataset = Dataset(split="train")
    val_dataset = Dataset(split="val")
    test_dataset = Dataset(split="test")

    return [train_dataset, val_dataset, test_dataset]


class FakeDataset(boring_classes.RandomDataset, datasets.Dataset):
    """Fake prediction dataset."""

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        size: int = 32,
        length: int = 10,
        metadata_keys: List[str] | None = None,
        filenames: List[str] | None = None,
    ):
        """Initializes the dataset."""
        super().__init__(size=size, length=length)
        self._split = split
        self._metadata_keys = metadata_keys
        self._filenames = filenames

    def filename(self, index: int) -> str:
        """Returns the filename for the given index."""
        if self._filenames:
            # This simulates the case where where multiple items can correspond to the same file.
            # e.g. in WSI classification, multiple patches can belong to the same slide.
            return random.choice(self._filenames)
        else:
            return f"{self._split}-{index}"

    @override
    def __getitem__(self, index: int):
        data = boring_classes.RandomDataset.__getitem__(self, index)
        target = random.choice([0, 1])
        if self._metadata_keys:
            metadata = {key: random.choice([0, 1, 2]) for key in self._metadata_keys}
            return data, target, metadata
        else:
            return data, target
