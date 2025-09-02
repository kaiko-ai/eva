"""TextPredictionDataset tests."""

from pathlib import Path

import pytest

from eva.language.data.datasets.prediction import TextPredictionDataset
from eva.language.data.datasets.schemas import TransformsSchema
from eva.language.data.datasets.typings import PredictionSample
from eva.language.data.messages import UserMessage


@pytest.mark.parametrize(
    "file_format, expected_length",
    [("jsonl", 8), ("csv", 8), ("parquet", 8)],
)
def test_length(prediction_dataset: TextPredictionDataset, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(prediction_dataset) == expected_length


@pytest.mark.parametrize(
    "file_format, index",
    [
        ("jsonl", 0),
        ("jsonl", 5),
        ("csv", 0),
        ("parquet", 0),
    ],
)
def test_sample(prediction_dataset: TextPredictionDataset, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = prediction_dataset[index]

    assert isinstance(sample, PredictionSample)
    assert sample.prediction in ["yes", "no", "maybe"]
    # Target can be string or int depending on the file format
    assert sample.target in ["0", "1", "2", 0, 1, 2]

    assert isinstance(sample.text, list)
    assert len(sample.text) == 1
    assert isinstance(sample.text[0], UserMessage)
    assert "Question:" in sample.text[0].content

    assert isinstance(sample.metadata, dict)


@pytest.mark.parametrize("file_format", ["jsonl"])
def test_no_text_column(assets_path: Path, file_format: str) -> None:
    """Tests dataset without text column."""
    path = Path(assets_path) / "language" / "predictions" / f"pubmedqa.{file_format}"
    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="prediction",
        target_column="target",
        text_column=None,
    )
    dataset.setup()

    sample = dataset[0]
    assert sample.text is None


@pytest.mark.parametrize("file_format", ["jsonl"])
def test_with_split(assets_path: Path, file_format: str) -> None:
    """Tests dataset with split filtering."""
    path = Path(assets_path) / "language" / "predictions" / f"pubmedqa.{file_format}"
    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="prediction",
        target_column="target",
        text_column="text",
        split="val",
    )
    dataset.setup()

    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample, PredictionSample)


@pytest.mark.parametrize("file_format", ["jsonl"])
def test_with_metadata_columns(assets_path: Path, file_format: str) -> None:
    """Tests dataset with metadata columns."""
    path = Path(assets_path) / "language" / "predictions" / f"pubmedqa.{file_format}"
    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="prediction",
        target_column="target",
        text_column="text",
        metadata_columns=["split"],
    )
    dataset.setup()

    sample = dataset[0]
    assert isinstance(sample.metadata, dict)
    assert "split" in sample.metadata
    assert sample.metadata["split"] == "val"


@pytest.mark.parametrize("file_format", ["jsonl"])
def test_unsupported_format(assets_path: Path, file_format: str) -> None:
    """Tests configuration with unsupported file format."""
    dataset = TextPredictionDataset(
        path="dummy.txt",
        prediction_column="prediction",
        target_column="target",
    )
    with pytest.raises(ValueError, match="Unsupported file extension"):
        dataset.configure()


@pytest.mark.parametrize("file_format", ["jsonl"])
def test_missing_columns(assets_path: Path, file_format: str) -> None:
    """Tests validation with missing columns."""
    path = Path(assets_path) / "language" / "predictions" / f"pubmedqa.{file_format}"

    # Missing prediction column
    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="non_existent",
        target_column="target",
    )
    dataset.configure()
    with pytest.raises(ValueError, match="Label column 'non_existent' not found"):
        dataset.validate()

    # Missing target column
    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="prediction",
        target_column="non_existent",
    )
    dataset.configure()
    with pytest.raises(ValueError, match="Label column 'non_existent' not found"):
        dataset.validate()


@pytest.mark.parametrize("file_format", ["jsonl"])
def test_with_transforms(assets_path: Path, file_format: str) -> None:
    """Tests dataset with transforms."""
    path = Path(assets_path) / "language" / "predictions" / f"pubmedqa.{file_format}"

    transforms = TransformsSchema(
        prediction=lambda x: f"transformed_{x}",
        target=lambda x: int(x) + 100,
    )

    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="prediction",
        target_column="target",
        text_column="text",
        transforms=transforms,
    )
    dataset.setup()

    sample = dataset[0]
    assert sample.prediction.startswith("transformed_")
    assert sample.target == 101


@pytest.fixture(scope="function")
def prediction_dataset(assets_path: Path, file_format: str) -> TextPredictionDataset:
    """TextPredictionDataset fixture."""
    path = Path(assets_path) / "language" / "predictions" / f"pubmedqa.{file_format}"
    dataset = TextPredictionDataset(
        path=str(path),
        prediction_column="prediction",
        target_column="target",
        text_column="text",
    )
    dataset.setup()
    return dataset
