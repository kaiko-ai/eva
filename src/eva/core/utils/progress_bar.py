"""Progress bar utility functions."""

import os

from tqdm import tqdm as _tqdm


def tqdm(*args, **kwargs) -> _tqdm:
    """Wrapper function for `tqdm.tqdm`."""
    refresh_rate = os.environ.get("TQDM_REFRESH_RATE")
    refresh_rate = int(refresh_rate) if refresh_rate is not None else None
    disable = bool(int(os.environ.get("TQDM_DISABLE", 0))) or (refresh_rate == 0)
    kwargs.setdefault("disable", disable)
    kwargs.setdefault("miniters", refresh_rate)
    return _tqdm(*args, **kwargs)
