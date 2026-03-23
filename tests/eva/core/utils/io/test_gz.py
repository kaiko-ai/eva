"""Tests for .gz file utilities."""

import os
import shutil

import pytest

from eva.core.utils.io import gz


@pytest.mark.parametrize(
    "subdir, keep",
    [
        (None, True),
        ("test_subdir", True),
        (None, False),
    ],
)
def test_gunzip(tmp_path: str, gzip_file: str, subdir: str | None, keep: bool) -> None:
    """Verifies proper extraction of gzip file contents."""
    unpack_dir = os.path.join(tmp_path, subdir) if subdir else tmp_path
    tmp_gzip_path = os.path.join(tmp_path, os.path.basename(gzip_file))
    shutil.copy(gzip_file, tmp_gzip_path)
    gz.gunzip_file(tmp_gzip_path, unpack_dir=unpack_dir, keep=keep)

    uncompressed_path = os.path.join(unpack_dir, "test.txt")
    assert os.path.isfile(uncompressed_path)
    with open(uncompressed_path, "r") as f:
        assert f.read() == "gz file test"

    if keep:
        assert os.path.isfile(tmp_gzip_path)
    else:
        assert not os.path.isfile(tmp_gzip_path)


@pytest.fixture()
def gzip_file(assets_path: str) -> str:
    """Provides the path to the test gzip file asset."""
    return os.path.join(assets_path, "core/archives/test.txt.gz")
