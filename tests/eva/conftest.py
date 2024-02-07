"""Shared configuration and fixtures for unit tests."""

import os

import pytest
from lightning_fabric.utilities import seed

TESTS_ROOT = os.path.dirname(__file__)
"""The tests directory."""

LIB_ROOT = os.path.dirname(os.path.dirname(TESTS_ROOT))
"""The test root directory full path."""

os.environ["TESTS_ROOT"] = TESTS_ROOT
"""Export tests directory to comply with the paths in the configs."""

seed.seed_everything(seed=42)
"""Sets the random seed."""


@pytest.fixture
def assets_path() -> str:
    """Provides the full path to test assets."""
    return os.path.join(TESTS_ROOT, "assets")


@pytest.fixture
def lib_path() -> str:
    """Provides the full path to the library source code."""
    return LIB_ROOT
