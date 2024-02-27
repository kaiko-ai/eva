"""Helper functions to record the evaluation run config and results."""

import hashlib
import json
import os
import sys
from datetime import datetime
from loguru import logger
from pathlib import Path

from loguru import logger


def _get_config_path() -> str | None:
    """Retrieve the config path from command line arguments.

    Returns:
        The path to the configuration file.
    """
    config_paths = [f for f in sys.argv if f.endswith(".yaml")]
    if len(config_paths) == 0:
        logger.info("No config file found from command line arguments")
        return None
    if len(config_paths) != 1:
        raise ValueError("Exactly one config file is allowed")
    return config_paths[0]


def _get_hash_from_config(config_path: str, max_hash_len: int) -> str:
    """Return a hash from the contents of the configuration file.

    Args:
        config_path: Path to the configuration file.
        max_hash_len: Maximum length of the returned hash.

    Returns:
        Hash of the configuration file content.
    """
    with open(config_path, "r") as stream:
        hash_id = hashlib.sha256(stream.read().encode("utf-8")).hexdigest()[:max_hash_len]
    return hash_id


def _save_result(results: dict, results_path: str) -> None:
    """Saves the results to a JSON file.

    Args:
        results: Dictionary containing the results.
        results_dir: Directory where the results should be saved.
    """
    if not os.path.isdir(Path(results_path).parent):
        os.makedirs(Path(results_path).parent)

    # results_path = os.path.join(results_dir, "results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, sort_keys=False)
        logger.info(f"Saved evaluation results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save job results to {results_path}: {e}")


def get_evaluation_id(start_time: datetime = datetime.now(), max_hash_len: int = 8) -> str:
    """Generates and returns a unique ID for the evaluation.
    The ID is composed of a timestamp and a hash.

    Args:
        start_time: The start time of the evaluation.
        max_hash_len: Maximum length of the returned hash.

    Returns:
        A string containing a timestamp and a hash.
    """
    config_path = _get_config_path()
    hash_id = _get_hash_from_config(config_path, max_hash_len) if config_path else ""

    timestamp = start_time.strftime("%Y%m%d-%H%M%S%f")
    return f"{timestamp}_{hash_id}"


def record_results(evaluation_results: dict, results_path: str, start_time, end_time) -> None:
    """Records the evaluation results to a JSON file.

    Args:
        evaluation_results: Dictionary containing the evaluation results.
        results_dir: Directory where the results should be saved.
        start_time: The start time of the evaluation.
        end_time: The end time of the evaluation.
    """
    results_dict = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": (end_time - start_time).total_seconds(),
        "metrics": evaluation_results,
    }
    _save_result(results_dict, results_path)
