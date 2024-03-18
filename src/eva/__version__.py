"""Fetches the version of the library."""

from importlib import metadata


def _fetch_version(package_name: str) -> str:
    """Fetches the version of an installed package.

    If it fails to do so, it returns a "*", indicating
    that the package has been installed as editable.

    Args:
        package_name: The name of the package to fetch
            the version of.

    Returns:
        A string representing the version of the library.
    """
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "*"


__version__ = _fetch_version("eva")
