"""Tests for Volume tv_tensor."""

import torch

from eva.vision.data.tv_tensors import Volume


def test_volume_creation() -> None:
    """Tests basic Volume creation."""
    data = torch.randn(2, 1, 16, 16)
    volume = Volume(data)

    assert isinstance(volume, Volume)
    assert volume.shape == (2, 1, 16, 16)


def test_volume_with_affine() -> None:
    """Tests Volume creation with affine matrix."""
    data = torch.randn(2, 1, 16, 16)
    affine = torch.eye(4)
    volume = Volume(data, affine=affine)

    assert volume.affine is not None
    torch.testing.assert_close(volume.affine, affine)


def test_volume_with_metadata() -> None:
    """Tests Volume creation with metadata."""
    data = torch.randn(2, 1, 16, 16)
    metadata = {"spacing": (1.0, 1.0, 1.0), "origin": (0.0, 0.0, 0.0)}
    volume = Volume(data, metadata=metadata)

    assert volume.metadata == metadata


def test_volume_dtype() -> None:
    """Tests Volume creation with specific dtype."""
    data = torch.randn(2, 1, 16, 16)
    volume = Volume(data, dtype=torch.float32)

    assert volume.dtype == torch.float32


def test_volume_requires_grad() -> None:
    """Tests Volume creation with requires_grad."""
    data = torch.randn(2, 1, 16, 16)
    volume = Volume(data, requires_grad=True)

    assert volume.requires_grad is True


def test_from_and_to_meta_tensor() -> None:
    """Tests roundtrip conversion between Volume and MetaTensor."""
    data = torch.randn(2, 1, 16, 16)
    affine = torch.eye(4, dtype=torch.float64)
    metadata = {"spacing": (1.0, 1.0, 1.0)}

    volume = Volume(data, affine=affine, metadata=metadata)
    meta_tensor = volume.to_meta_tensor()
    volume_back = Volume.from_meta_tensor(meta_tensor)

    torch.testing.assert_close(volume_back.data, volume.data)
    torch.testing.assert_close(volume_back.affine, affine)
    assert volume_back.metadata is not None
    assert volume_back.metadata["spacing"] == (1.0, 1.0, 1.0)


def test_multiple_volumes() -> None:
    """Tests creation of multiple Volume instances and if affine and metadata are preserved."""
    vol1 = Volume(torch.randn(1, 10, 64, 64), affine=torch.eye(4) * 1, metadata={"id": "volume1"})
    vol2 = Volume(torch.randn(1, 10, 64, 64), affine=torch.eye(4) * 2, metadata={"id": "volume2"})

    assert vol1.affine is not None and vol2.affine is not None
    assert vol1.affine[0, 0] != vol2.affine[0, 0]

    assert vol1.metadata["id"] == "volume1"
    assert vol2.metadata["id"] == "volume2"
