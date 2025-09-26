"""Utility functions for handling paths."""

import os


def home_dir():
    """Get eva's home directory for caching."""
    torch_home = os.path.expanduser(
        os.getenv(
            "EVA_HOME",
            os.path.join("~/.cache", "eva"),
        )
    )
    return torch_home
