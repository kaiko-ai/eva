"""Tests the datamodule schemas."""

import pytest

from eva.core.data.datamodules import schemas


def test_datasets_schema(dataset_schema: schemas.DatasetsSchema) -> None:
    """Tests the trainer dataset schema."""
    assert hasattr(dataset_schema, "train")
    assert hasattr(dataset_schema, "val")
    assert hasattr(dataset_schema, "test")
    assert hasattr(dataset_schema, "predict")


def test_dataloader_schema(dataloader_schema: schemas.DataloadersSchema) -> None:
    """Tests the trainer dataset schema."""
    assert hasattr(dataloader_schema, "train")
    assert hasattr(dataloader_schema, "val")
    assert hasattr(dataloader_schema, "test")
    assert hasattr(dataloader_schema, "predict")


@pytest.fixture(scope="function")
def dataset_schema() -> schemas.DatasetsSchema:
    """Creates the default dataset schema."""
    return schemas.DatasetsSchema()


@pytest.fixture(scope="function")
def dataloader_schema() -> schemas.DataloadersSchema:
    """Creates the default dataloader schema."""
    return schemas.DataloadersSchema()
