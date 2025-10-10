"""QuiltVQA dataset tests."""

from unittest import mock

import pytest
from datasets import Dataset
from PIL import Image
from torchvision import tv_tensors

from eva.language.data.messages import Message
from eva.multimodal.data import datasets


@pytest.mark.parametrize(
    "split, expected_length",
    [("test", 985), (None, 985)],
)
def test_length(quiltvqa_dataset: datasets.QuiltVQA, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(quiltvqa_dataset) == expected_length


@pytest.mark.parametrize(
    "split, index",
    [
        ("test", 0),
        ("test", 10),
        (None, 0),
    ],
)
def test_sample(quiltvqa_dataset: datasets.QuiltVQA, index: int) -> None:
    """Tests the format of a dataset sample."""
    sample = quiltvqa_dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 4

    text, image, target, metadata = sample
    assert isinstance(text, list)
    assert all(isinstance(item, Message) for item in text)

    content = text[0].content
    assert content.startswith(quiltvqa_dataset.prompt_render_kwargs["preamble"])
    assert "Question:" in content

    assert isinstance(image, tv_tensors.Image)
    assert image.ndim == 3

    assert isinstance(target, str)

    assert isinstance(metadata, dict)
    required_keys = {"answer_type", "context"}
    assert all(key in metadata for key in required_keys)


@pytest.mark.parametrize("split", [None])
def test_load_image(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests loading an image from the dataset."""
    image = quiltvqa_dataset.load_image(0)
    assert isinstance(image, tv_tensors.Image)
    assert image.ndim == 3


@pytest.mark.parametrize("split", [None])
def test_load_text(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests loading text from the dataset."""
    messages = quiltvqa_dataset.load_text(0)
    assert isinstance(messages, list)
    assert all(isinstance(item, Message) for item in messages)
    assert len(messages) > 0


@pytest.mark.parametrize("split", [None])
def test_load_target(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests loading target from the dataset."""
    target = quiltvqa_dataset.load_target(0)
    assert isinstance(target, str)


@pytest.mark.parametrize("split", [None])
def test_load_metadata(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests loading metadata from the dataset."""
    metadata = quiltvqa_dataset.load_metadata(0)
    assert isinstance(metadata, dict)
    assert "answer_type" in metadata
    assert "context" in metadata


@pytest.mark.parametrize("split", [None])
def test_prepare_data_no_root(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests dataset preparation without specifying a root directory."""
    assert isinstance(quiltvqa_dataset.dataset, Dataset)
    assert len(quiltvqa_dataset) > 0


@pytest.mark.parametrize("split", [None])
def test_prepare_data_without_download(split) -> None:
    """Tests dataset preparation when download is disabled and cache is missing."""
    dataset = datasets.QuiltVQA(split=split, download=False)

    with pytest.raises(ValueError, match="Dataset path not found."):
        dataset.prepare_data()


@pytest.mark.parametrize("split", [None])
def test_invalid_split(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests that invalid splits raise an error."""
    quiltvqa_dataset._split = "train"

    with pytest.raises(ValueError, match="Available splits are"):
        quiltvqa_dataset.validate()


@pytest.mark.parametrize("split", [None])
def test_index_out_of_range(quiltvqa_dataset: datasets.QuiltVQA) -> None:
    """Tests that out of range indices raise an error."""
    with pytest.raises(IndexError):
        quiltvqa_dataset.load_text(10000)

    with pytest.raises(IndexError):
        quiltvqa_dataset.load_target(-1)


@pytest.fixture(scope="function")
def quiltvqa_dataset(split: None) -> datasets.QuiltVQA:
    """QuiltVQA dataset fixture with mocked download."""
    # Create a mock dataset to avoid actual download
    mock_dataset = Dataset.from_dict(
        {
            "question": ["What is shown in this image?"] * 985,
            "answer": ["A medical image"] * 985,
            "image": [Image.new("RGB", (224, 224))] * 985,
            "answer_type": ["descriptive"] * 985,
            "context": ["medical context"] * 985,
        }
    )

    dataset = datasets.QuiltVQA(split=split, download=False)

    # Mock the _load_dataset method to return our mock dataset
    with mock.patch.object(dataset, "_load_dataset", return_value=mock_dataset):
        dataset.prepare_data()

    return dataset
