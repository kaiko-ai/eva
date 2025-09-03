"""Tests for MultimodalDataset base class."""

from typing_extensions import override

from eva.multimodal.data.datasets.base import MultimodalDataset


class ConcreteMultimodalDataset(MultimodalDataset):
    """Concrete implementation for testing."""

    def __init__(self):
        """Initialize test dataset."""
        self._data = ["sample1", "sample2", "sample3"]

    @override
    def __len__(self):
        return len(self._data)

    @override
    def __getitem__(self, index):
        return self._data[index]


def test_multimodal_dataset_inheritance():
    """Test that MultimodalDataset can be inherited and used."""
    dataset = ConcreteMultimodalDataset()

    assert len(dataset) == 3
    assert dataset[0] == "sample1"
    assert dataset[1] == "sample2"
    assert dataset[2] == "sample3"
