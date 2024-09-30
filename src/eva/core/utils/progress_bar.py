"""Progress bar utility functions."""

import os
import sys

from tqdm import tqdm as _tqdm


def tqdm(*args, **kwargs) -> _tqdm:
    """Wrapper function for `tqdm.tqdm`."""
    refresh_rate = int(os.environ.get("TQDM_REFRESH_RATE", 1))
    try:
        disable = bool(int(os.environ["TQDM_DISABLE"])) or refresh_rate == 0
    except KeyError:
        disable = not sys.stdout.isatty()
    except (ValueError, TypeError):
        disable = False
    kwargs.setdefault("disable", disable)
    kwargs.setdefault("miniters", max(1, refresh_rate))
    return _tqdm(*args, **kwargs)
