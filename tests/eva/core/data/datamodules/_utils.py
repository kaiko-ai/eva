"""Utilities and helper functions for the datamodule tests."""

from typing_extensions import override

from eva.core.data import datasets as datasets_lib


class DummyDataset(datasets_lib.Dataset):
    """Dummy dataset."""

    def __init__(self) -> None:
        super().__init__()

        self._prepare_data_called = False
        self._setup_called = False
        self._teardown_called = False

    def prepare_data(self) -> None:
        """Prepare dataset method."""
        self._prepare_data_called = True

    def setup(self) -> None:
        """Setup dataset method."""
        self._setup_called = True

    def teardown(self) -> None:
        """Teardown dataset method."""
        self._teardown_called = True

    @override
    def __getitem__(self, index: int) -> int:
        return 1

    def __len__(self) -> int:
        """Returns total number of samples of the dataset."""
        return 100
