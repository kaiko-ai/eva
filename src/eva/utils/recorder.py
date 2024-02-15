"""Helper functions to record the evaluation run config and results"""

import hashlib
import json
import os
import sys
from datetime import datetime

from loguru import logger


def _get_config_path() -> str:
    """Retrieve the config path from command line arguments.

    Returns:
        The path to the configuration file.
    """
    config_paths = [f for f in sys.argv if f.endswith(".yaml")]
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


def _copy_config_file(source_path: str, destination_path: str) -> None:
    """Copies a file from source_path to destination_path.

    Args:
        source_path: Path to the source file.
        destination_path: Path to the destination file.
    """
    try:
        with open(source_path, "r") as f1:
            with open(destination_path, "w") as f2:
                data = f1.read()
                if not data:
                    logger.warning(f"File {source_path} is empty")
                else:
                    f2.write(data)
        logger.info(f"Saved a copy of the run config to {destination_path}")

    except Exception as e:
        logger.error(f"Failed to copy {source_path} to {destination_path}: {e}")


def _save_result(results: dict, results_dir: str) -> None:
    """Saves the results to a JSON file.

    Args:
        results: Dictionary containing the results.
        results_dir: Directory where the results should be saved.
    """
    results_path = os.path.join(results_dir, "results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, sort_keys=False)
        logger.info(f"Saved evaluation results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save job results to {results_path}: {e}")


def get_run_id(start_time: datetime, max_hash_len: int = 8) -> str:
    """Generates and returns a unique ID for the evaluation. The ID is composed of a
        timestamp and a hash.

    Args:
        max_hash_len: Maximum length of the returned hash.

    Returns:
        A string containing a timestamp and a hash.
    """
    config_path = _get_config_path()
    hash_id = _get_hash_from_config(config_path, max_hash_len)

    timestamp = start_time.strftime("%Y%m%d-%H%M%S%f")
    return f"{timestamp}_{hash_id}"


def record_results(evaluation_results: dict, results_dir: str, start_time, end_time) -> None:
    """Records the evaluation results to a JSON file.

    Args:
        evaluation_results: Dictionary containing the evaluation results.
        results_dir: Directory where the results should be saved.
        start_time: The start time of the evaluation.
        end_time: The end time of the evaluation.
    """
    config_path = _get_config_path()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    _copy_config_file(config_path, os.path.join(results_dir, "run_config.yaml"))
    results_dict = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": (end_time - start_time).total_seconds(),
        "metrics": evaluation_results,
    }
    _save_result(results_dict, results_dir)
