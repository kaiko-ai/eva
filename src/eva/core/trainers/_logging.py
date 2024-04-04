"""Helper functions and utilities for trainer logging."""

import hashlib
import sys
from datetime import datetime

from lightning_fabric.utilities import cloud_io
from loguru import logger


def generate_session_id() -> str:
    """Generates and returns a unique string ID of an experiment.

    The ID is composed of the run timestamp and a hash based on th used
    config. If the configuration hash is an empty string, it will use
    only the timestamp.
    """
    timestamp = _generate_timestamp_hash()
    config_hash = _generate_config_hash()
    return f"{timestamp}_{config_hash}" if config_hash else timestamp


def _generate_timestamp_hash() -> str:
    """Generate a time-based hash id."""
    timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d-%H%M%S%f")


def _generate_config_hash(max_hash_len: int = 8) -> str:
    """Generates a hash id based on a yaml configuration file.

    Args:
        max_hash_len: The maximum length of the produced hash id.
    """
    config_path = _fetch_config_path()
    if config_path is None:
        logger.warning(
            "No or multiple configuration files found from command line arguments. "
            "No configuration hash code will be created for this experiment."
        )
        return ""

    return _generate_hash_from_config(config_path, max_hash_len)


def _fetch_config_path() -> str | None:
    """Retrieves the configuration path from command line arguments.

    It returns `None` if no or multiple configuration files found in
    the system arguments.

    Returns:
        The path to the configuration file.
    """
    inputs = sys.argv
    config_paths = [inputs[i + 1] for i, arg in enumerate(inputs) if arg == "--config"]
    if len(config_paths) == 0 or len(config_paths) > 1:
        # TODO combine the multiple configuration files
        # and produced hash for the merged one.
        return None

    return config_paths[0]


def _generate_hash_from_config(path: str, max_hash_len: int = 8) -> str:
    """Return a hash from the contents of the configuration file.

    Args:
        path: Path to the configuration file.
        max_hash_len: Maximum length of the returned hash.

    Returns:
        Hash of the configuration file content.
    """
    fs = cloud_io.get_filesystem(path)
    with fs.open(path, "r") as stream:
        config = stream.read()
        if isinstance(config, str):
            config = config.encode("utf-8")
        config_sha256 = hashlib.sha256(config)
        hash_id = config_sha256.hexdigest()
    return hash_id[:max_hash_len]
