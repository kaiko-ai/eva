"""Utility functions related to package requirements."""

import importlib
from typing import Dict

import packaging.version


def fetch_version(name: str) -> str | None:
    """Fetch the installed version of a package.

    Args:
        name: The name of the package.

    Returns:
        A string representing the installed version of the package, or None if not found.
    """
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", None)
    except ImportError:
        return None


def below(name: str, version: str) -> bool:
    """Check if the installed version of a package is below a certain version.

    Args:
        name: The name of the package.
        version: The version to compare against.

    Returns:
        True if the installed version is below the specified version, False otherwise.
    """
    actual = fetch_version(name)
    if actual:
        return packaging.version.parse(actual) < packaging.version.parse(version)
    return False


def above_or_equal(name: str, version: str) -> bool:
    """Check if the installed version of a package is above a certain version.

    Args:
        name: The name of the package.
        version: The version to compare against.

    Returns:
        True if the installed version is above the specified version, False otherwise.
    """
    actual = fetch_version(name)
    if actual:
        return packaging.version.parse(actual) >= packaging.version.parse(version)
    return False


def check_min_versions(requirements: Dict[str, str]) -> None:
    """Check installed package versions against requirements dict.

    Args:
        requirements: A dictionary where keys are package names and
            values are minimum required versions.

    Raises:
        ImportError: If any package does not meet the minimum required version.
    """
    for package, min_version in requirements.items():
        if below(package, min_version):
            raise ImportError(
                f"Package '{package}' version {fetch_version(package)} does not meet "
                f"the minimum required version {min_version}."
            )
