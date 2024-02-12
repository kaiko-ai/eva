"""Data classes and structures for datasets."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class DownloadResource:
    """Contains download information for a specific resource."""

    filename: str
    """The filename of the resource."""

    url: str
    """The URL of the resource."""

    md5: str | None = None
    """The MD5 hash of the resource."""


@dataclasses.dataclass(frozen=True)
class SplitRatios:
    """Contains split ratios for train, val and test."""

    train: float
    """Train split ratio."""

    val: float
    """Validation split ratio."""

    test: float
    """Test split ratio."""
