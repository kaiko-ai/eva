"""PathMMU Atlas dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.language.data.messages import UserMessage
from eva.multimodal.data.datasets.multiple_choice import path_mmu_atlas
from eva.multimodal.data.datasets.typings import TextImageSample


@pytest.fixture(scope="function")
def path_mmu_atlas_dataset(
    split: Literal["val", "test", "test_tiny"], assets_path: str
) -> path_mmu_atlas.PathMMUAtlas:
    """PathMMU Atlas dataset fixture."""
    dataset = path_mmu_atlas.PathMMUAtlas(
        root=os.path.join(assets_path, "multimodal", "datasets", "path_mmu_atlas"),
        split=split,
    )
    dataset.prepare_data()
    dataset.configure()
    return dataset


@pytest.mark.parametrize(
    "split, expected_length",
    [("val", 2), ("test", 3), ("test_tiny", 2)],
)
def test_length(path_mmu_atlas_dataset: path_mmu_atlas.PathMMUAtlas, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(path_mmu_atlas_dataset) == expected_length


@pytest.mark.parametrize(
    "split",
    ["val", "test", "test_tiny"],
)
def test_sample(path_mmu_atlas_dataset: path_mmu_atlas.PathMMUAtlas) -> None:
    """Tests the format of a dataset sample."""
    sample = path_mmu_atlas_dataset[0]
    assert isinstance(sample, TextImageSample)

    # Test text component
    assert isinstance(sample.text, list)
    assert len(sample.text) == 1
    assert isinstance(sample.text[0], UserMessage)
    content = str(sample.text[0].content)
    assert "Question:" in content

    # Test images component
    assert isinstance(sample.images, list)
    assert len(sample.images) == 1
    assert isinstance(sample.images[0], tv_tensors.Image)
    assert sample.images[0].ndim == 3

    # Test target - should be an integer index (0=A, 1=B, 2=C, 3=D, 4=E)
    assert isinstance(sample.target, int)
    assert sample.target in [0, 1, 2, 3, 4]

    # Test metadata
    assert sample.metadata is not None
    assert "answer" in sample.metadata
    assert "explanation" in sample.metadata
    assert "source_img" in sample.metadata


@pytest.mark.parametrize(
    "split",
    ["val", "test", "test_tiny"],
)
def test_load_images(path_mmu_atlas_dataset: path_mmu_atlas.PathMMUAtlas) -> None:
    """Tests loading images from the dataset."""
    images = path_mmu_atlas_dataset.load_images(0)
    assert isinstance(images, list)
    assert len(images) == 1
    assert isinstance(images[0], tv_tensors.Image)
    assert images[0].ndim == 3


@pytest.mark.parametrize(
    "split",
    ["val", "test", "test_tiny"],
)
def test_load_text(path_mmu_atlas_dataset: path_mmu_atlas.PathMMUAtlas) -> None:
    """Tests loading text from the dataset."""
    messages = path_mmu_atlas_dataset.load_text(0)
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert isinstance(messages[0], UserMessage)


@pytest.mark.parametrize(
    "split",
    ["val", "test", "test_tiny"],
)
def test_load_target(path_mmu_atlas_dataset: path_mmu_atlas.PathMMUAtlas) -> None:
    """Tests loading target from the dataset."""
    target = path_mmu_atlas_dataset.load_target(0)
    assert isinstance(target, int)
    assert target in [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "split",
    ["val", "test", "test_tiny"],
)
def test_load_metadata(path_mmu_atlas_dataset: path_mmu_atlas.PathMMUAtlas) -> None:
    """Tests loading metadata from the dataset."""
    metadata = path_mmu_atlas_dataset.load_metadata(0)
    assert isinstance(metadata, dict)
    assert "answer" in metadata
    assert "explanation" in metadata
    assert "source_img" in metadata


def test_class_to_idx(assets_path: str) -> None:
    """Tests the class_to_idx mapping."""
    dataset = path_mmu_atlas.PathMMUAtlas(
        root=os.path.join(assets_path, "multimodal", "datasets", "path_mmu_atlas"),
        split="val",
    )
    assert dataset.class_to_idx == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def test_classes(assets_path: str) -> None:
    """Tests the classes property."""
    dataset = path_mmu_atlas.PathMMUAtlas(
        root=os.path.join(assets_path, "multimodal", "datasets", "path_mmu_atlas"),
        split="val",
    )
    assert dataset.classes == ["A", "B", "C", "D", "E"]


def test_five_option_question(assets_path: str) -> None:
    """Tests that questions with 5 options (A-E) are handled correctly."""
    dataset = path_mmu_atlas.PathMMUAtlas(
        root=os.path.join(assets_path, "multimodal", "datasets", "path_mmu_atlas"),
        split="test_tiny",
    )
    dataset.prepare_data()
    dataset.configure()

    # Check that all targets are valid indices (0-4 for A-E)
    targets = [dataset.load_target(i) for i in range(len(dataset))]
    assert all(t in [0, 1, 2, 3, 4] for t in targets)
