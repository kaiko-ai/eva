"""Language Dataset base class."""

import abc
from typing import Generic, TypeVar

from eva.core.data.datasets import base

DataSample = TypeVar("DataSample")
"""The data sample type."""


class LanguageDataset(base.MapDataset, abc.ABC, Generic[DataSample]):
    """Base dataset class for text tasks."""
