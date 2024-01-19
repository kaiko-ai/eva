"""Url conversion utilities."""
import functools

import adlfs
from fsspec.implementations import local


@functools.singledispatch
def path_to_url(
    fs,  # type: ignore
    path: str,
) -> str:
    """Converts a path to a fully qualified url.

    This addresses https://github.com/fsspec/filesystem_spec/issues/1169

    Args:
        fs: The desired filesystem class.
        path: The path to convert.
    """
    raise NotImplementedError(
        f"path_to_url for fs '{fs.__class__.__name__}' is not yet implemented."
    )


@path_to_url.register
def _(fs: local.LocalFileSystem, path: str) -> str:
    return path


@path_to_url.register
def _(fs: adlfs.spec.AzureBlobFileSystem, path: str) -> str:
    """Adds scalar to a list of supported loggers."""
    container_name = path.split("/")[0]
    path = "/".join(path.split("/")[1:])
    return f"{fs.protocol}://{container_name}@{fs.account_name}.blob.core.windows.net/" + path
