"""Helper dataclasses and data structures for vision datasets."""

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
