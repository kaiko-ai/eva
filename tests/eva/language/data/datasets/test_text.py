"""Tests for TextDataset class."""

from typing import Any, Dict

from typing_extensions import override

from eva.language.data.datasets.schemas import TransformsSchema
from eva.language.data.datasets.text import TextDataset
from eva.language.data.datasets.typings import TextSample
from eva.language.data.messages import UserMessage


class ConcreteTextDataset(TextDataset):
    """Concrete implementation for testing."""

    def __init__(self, transforms: TransformsSchema | None = None):
        """Initialize test dataset."""
        super().__init__(transforms=transforms)
        self._size = 3

    @override
    def __len__(self) -> int:
        return self._size

    @override
    def load_text(self, index: int):
        return [UserMessage(content=f"Text {index}")]

    @override
    def load_target(self, index: int) -> int:
        return index

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return {"index": index}


def test_text_dataset_getitem():
    """Test __getitem__ returns proper TextSample."""
    dataset = ConcreteTextDataset()
    sample = dataset[0]

    assert isinstance(sample, TextSample)
    assert len(sample.text) == 1
    assert sample.text[0].content == "Text 0"
    assert sample.target == 0
    assert sample.metadata == {"index": 0}


def test_text_dataset_with_transforms():
    """Test dataset applies transforms correctly."""

    def text_transform(text):
        # Modify the text content
        return [UserMessage(content=text[0].content.upper())]

    def target_transform(target):
        return target * 10

    transforms = TransformsSchema(
        text=text_transform,
        target=target_transform,
    )

    dataset = ConcreteTextDataset(transforms=transforms)
    sample = dataset[1]

    assert sample.text[0].content == "TEXT 1"
    assert sample.target == 10
    assert sample.metadata == {"index": 1}


def test_text_dataset_without_transforms():
    """Test dataset without transforms returns original data."""
    dataset = ConcreteTextDataset(transforms=None)
    sample = dataset[2]

    assert sample.text[0].content == "Text 2"
    assert sample.target == 2
    assert sample.metadata == {"index": 2}
