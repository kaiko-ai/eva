"""Vision dataset typings API."""

import enum


class DatasetType(enum.Enum):
    """Type of the dataset."""

    PATCH = 0
    SLIDE = 1
