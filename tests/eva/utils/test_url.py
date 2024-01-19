"""Tests for path_to_url."""
from fsspec.core import url_to_fs

from eva.utils import url


def test_url_to_fs_local():
    """Tests for local."""
    original_url = "/some/path"
    fs, path = url_to_fs(original_url)
    result_url = url.path_to_url(fs, path)
    assert original_url == result_url
    original_url = "some/path"
    fs, path = url_to_fs(original_url)
    result_url = url.path_to_url(fs, path)
    assert original_url == result_url[-len(original_url) :]


def test_url_to_fs_adlfs():
    """Tests for adlfs."""
    original_url = "abfs://some_container@kaiko.blob.core.windows.net/some/path/"
    fs, path = url_to_fs(original_url)
    result_url = url.path_to_url(fs, path)
    assert original_url == result_url
