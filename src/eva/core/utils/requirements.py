"""Utility functions related to package requirements."""

import importlib
from typing import Dict

from packaging import version


def check_dependencies(requirements: Dict[str, str]) -> None:
    """Check installed package versions against requirements dict.

    Args:
        requirements: A dictionary where keys are package names and
            values are minimum required versions.

    Raises:
        ImportError: If any package does not meet the minimum required version.
    """
    for package, min_version in requirements.items():
        module = importlib.import_module(package)
        actual = getattr(module, "__version__", None)
        if actual and not (version.parse(actual) >= version.parse(min_version)):
            raise ImportError(
                f"Package '{package}' version {actual} does not meet "
                f"the minimum required version {min_version}."
            )
