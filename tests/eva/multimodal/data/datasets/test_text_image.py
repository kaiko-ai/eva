"""Tests for TextImageDataset class."""

from typing import Any, Dict

import torch
from torchvision import tv_tensors
from typing_extensions import override

from eva.language.data.messages import UserMessage
from eva.multimodal.data.datasets.schemas import TransformsSchema
from eva.multimodal.data.datasets.text_image import TextImageDataset
from eva.multimodal.data.datasets.typings import TextImageSample


class ConcreteTextImageDataset(TextImageDataset):
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
    def load_image(self, index: int) -> tv_tensors.Image:
        return tv_tensors.Image(torch.rand(3, 224, 224))

    @override
    def load_target(self, index: int) -> int:
        return index

    @override
    def load_metadata(self, index: int) -> Dict[str, Any]:
        return {"index": index}


def test_text_image_dataset_getitem():
    """Test __getitem__ returns proper TextImageSample."""
    dataset = ConcreteTextImageDataset()
    sample = dataset[0]

    assert isinstance(sample, TextImageSample)
    assert len(sample.text) == 1
    assert sample.text[0].content == "Text 0"
    assert isinstance(sample.image, tv_tensors.Image)
    assert sample.target == 0
    assert sample.metadata == {"index": 0}


def test_text_image_dataset_with_transforms():
    """Test dataset applies transforms correctly."""

    def text_transform(text):
        # Modify the text content
        return [UserMessage(content=text[0].content.upper())]

    def image_transform(image):
        # Simple transform - just return a different sized image
        return tv_tensors.Image(torch.rand(3, 128, 128))

    def target_transform(target):
        return target * 10

    transforms = TransformsSchema(
        text=text_transform, image=image_transform, target=target_transform
    )

    dataset = ConcreteTextImageDataset(transforms=transforms)
    sample = dataset[1]

    assert sample.text[0].content == "TEXT 1"
    assert sample.image.shape == (3, 128, 128)
    assert sample.target == 10


def test_text_image_dataset_without_transforms():
    """Test dataset without transforms returns original data."""
    dataset = ConcreteTextImageDataset(transforms=None)
    sample = dataset[2]

    assert sample.text[0].content == "Text 2"
    assert sample.image.shape == (3, 224, 224)
    assert sample.target == 2
