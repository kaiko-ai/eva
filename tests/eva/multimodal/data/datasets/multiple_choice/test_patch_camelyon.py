"""PatchCamelyon multimodal dataset tests."""

import os
from typing import Literal

import pytest
from torchvision import tv_tensors

from eva.language.data.messages import Message, UserMessage
from eva.multimodal.data.datasets.multiple_choice import patch_camelyon
from eva.multimodal.data.datasets.typings import TextImageSample


@pytest.mark.parametrize(
    "split, expected_length",
    [("train", 4), ("val", 2), ("test", 1)],
)
def test_length(patch_camelyon_dataset: patch_camelyon.PatchCamelyon, expected_length: int) -> None:
    """Tests the length of the dataset."""
    assert len(patch_camelyon_dataset) == expected_length


@pytest.mark.parametrize(
    "split",
    ["train", "val", "test"],
)
def test_sample(patch_camelyon_dataset: patch_camelyon.PatchCamelyon) -> None:
    """Tests the format of a dataset sample."""
    sample = patch_camelyon_dataset[0]
    # assert data sample is a TextImageSample
    assert isinstance(sample, TextImageSample)

    # Test text component
    assert isinstance(sample.text, list)
    assert len(sample.text) == 1
    assert isinstance(sample.text[0], UserMessage)
    assert "metastatic breast tissue" in sample.text[0].content

    # Test image component
    assert isinstance(sample.image, tv_tensors.Image)
    assert sample.image.shape == (3, 96, 96)

    # Test target
    assert isinstance(sample.target, int)
    assert sample.target in [0, 1]

    # Test metadata
    assert sample.metadata is not None


@pytest.mark.parametrize(
    "split",
    ["train", "val", "test"],
)
def test_custom_prompt(split: Literal["train", "val", "test"], assets_path: str) -> None:
    """Tests the dataset with a custom prompt."""
    custom_prompt = "Is this image showing cancer? Answer: "
    dataset = patch_camelyon.PatchCamelyon(
        root=os.path.join(assets_path, "vision", "datasets", "patch_camelyon"),
        split=split,
        prompt=custom_prompt,
    )

    sample = dataset[0]
    assert isinstance(sample.text, list)
    assert isinstance(sample.text[0], Message)
    assert sample.text[0].content == custom_prompt


@pytest.mark.parametrize(
    "split",
    ["train", "val", "test"],
)
def test_max_samples(split: Literal["train", "val", "test"], assets_path: str) -> None:
    """Tests the dataset with max_samples limit."""
    max_samples = 1
    dataset = patch_camelyon.PatchCamelyon(
        root=os.path.join(assets_path, "vision", "datasets", "patch_camelyon"),
        split=split,
        max_samples=max_samples,
    )

    assert len(dataset) == max_samples


def test_class_to_idx(assets_path: str) -> None:
    """Tests the class_to_idx mapping."""
    dataset = patch_camelyon.PatchCamelyon(
        root=os.path.join(assets_path, "vision", "datasets", "patch_camelyon"),
        split="train",
    )
    assert dataset.class_to_idx == {"A": 0, "B": 1}


@pytest.fixture(scope="function")
def patch_camelyon_dataset(
    split: Literal["train", "val", "test"], assets_path: str
) -> patch_camelyon.PatchCamelyon:
    """PatchCamelyon multimodal dataset fixture."""
    dataset = patch_camelyon.PatchCamelyon(
        root=os.path.join(assets_path, "vision", "datasets", "patch_camelyon"),
        split=split,
    )
    return dataset
