"""Unit tests for requirements utility functions."""

import importlib
from unittest import mock

import pytest

from eva.core.utils import requirements


class MockModule:
    """Mock module for testing."""

    def __init__(self, version: str | None = None):
        """Creates an instance."""
        if version:
            self.__version__ = version


@pytest.mark.parametrize(
    "module_fixture,expected",
    [
        (MockModule(version="1.2.3"), "1.2.3"),
        (MockModule(), None),
    ],
)
def test_fetch_version_with_module(module_fixture, expected):
    """Test fetch_version with different module configurations."""
    with mock.patch.object(importlib, "import_module", return_value=module_fixture):
        result = requirements.fetch_version("test_module")
        assert result == expected


def test_fetch_version_import_error():
    """Test fetch_version when module cannot be imported."""
    with mock.patch.object(importlib, "import_module", side_effect=ImportError):
        result = requirements.fetch_version("non_existent_module")
        assert result is None


@pytest.mark.parametrize(
    "installed,required,expected",
    [
        ("1.0.0", "2.0.0", True),  # lower version
        ("2.0.0", "1.0.0", False),  # higher version
        ("1.0.0", "1.0.0", False),  # equal version
        (None, "1.0.0", False),  # no version
    ],
)
def test_below(installed, required, expected):
    """Test below function with various version scenarios."""
    with mock.patch.object(requirements, "fetch_version", return_value=installed):
        assert requirements.below("test_module", required) == expected


@pytest.mark.parametrize(
    "installed,required,expected",
    [
        ("2.0.0", "1.0.0", True),  # higher version
        ("1.0.0", "2.0.0", False),  # lower version
        ("1.0.0", "1.0.0", True),  # equal version
        (None, "1.0.0", False),  # no version
    ],
)
def test_above_or_equal(installed, required, expected):
    """Test above_or_equal function with various version scenarios."""
    with mock.patch.object(requirements, "fetch_version", return_value=installed):
        assert requirements.above_or_equal("test_module", required) == expected


def test_check_min_versions_all_satisfied():
    """Test check_min_versions when all requirements are satisfied."""
    reqs = {
        "package1": "1.0.0",
        "package2": "2.0.0",
    }

    with mock.patch.object(requirements, "fetch_version") as mock_fetch:
        mock_fetch.side_effect = ["1.5.0", "2.1.0"]
        with mock.patch.object(requirements, "below", return_value=False):
            requirements.check_min_versions(reqs)


def test_check_min_versions_not_satisfied():
    """Test check_min_versions when a requirement is not satisfied."""
    reqs = {
        "package1": "2.0.0",
        "package2": "3.0.0",
    }

    with mock.patch.object(requirements, "fetch_version", return_value="1.0.0"):
        with mock.patch.object(requirements, "below", return_value=True):
            with pytest.raises(ImportError) as exc_info:
                requirements.check_min_versions(reqs)

            assert "package1" in str(exc_info.value)
            assert "1.0.0" in str(exc_info.value)
            assert "2.0.0" in str(exc_info.value)


def test_check_min_versions_empty():
    """Test check_min_versions with empty requirements."""
    requirements.check_min_versions({})


@pytest.mark.parametrize(
    "installed,required,below_expected,above_expected",
    [
        ("1.0.0.dev0", "1.0.0", True, False),  # dev version
        ("1.0.0rc1", "1.0.0", True, False),  # rc version
        ("1.2.3.post1", "1.2.4", True, None),  # post version - below
    ],
)
def test_version_comparison_special_versions(installed, required, below_expected, above_expected):
    """Test version comparison with special version formats."""
    with mock.patch.object(requirements, "fetch_version", return_value=installed):
        assert requirements.below("test_module", required) == below_expected
        if above_expected is not None:
            assert requirements.above_or_equal("test_module", required) == above_expected


@pytest.mark.parametrize(
    "installed,required,expected",
    [
        ("1.2.3.post1", "1.2.3", True),  # post version - above_or_equal
    ],
)
def test_above_or_equal_special_versions(installed, required, expected):
    """Test above_or_equal with special version formats."""
    with mock.patch.object(requirements, "fetch_version", return_value=installed):
        assert requirements.above_or_equal("test_module", required) == expected
