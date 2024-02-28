"""Helper functions to record the evaluation run config and results."""

import hashlib
import json
import os
import sys
from datetime import datetime
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
        results_path: Directory where the results should be saved.
    """
    if not os.path.isdir(Path(results_path).parent):
        os.makedirs(Path(results_path).parent)

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, sort_keys=False)
        logger.info(f"Saved evaluation results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save job results to {results_path}: {e}")


def get_evaluation_id(start_time: datetime | None = None, max_hash_len: int = 8) -> str:
    """Generates and returns a unique ID for the evaluation.

    The ID is composed of a timestamp and a hash.

    Args:
        start_time: The start time of the evaluation.
        max_hash_len: Maximum length of the returned hash.

    Returns:
        A string containing a timestamp and a hash.
    """
    if start_time is None:
        start_time = datetime.now()

    config_path = _get_config_path()
    hash_id = _get_hash_from_config(config_path, max_hash_len) if config_path else ""

    timestamp = start_time.strftime("%Y%m%d-%H%M%S%f")
    return f"{timestamp}_{hash_id}"


def record_results(
    evaluation_results: dict, results_path: str, start_time: datetime, end_time: datetime
) -> None:
    """Records the evaluation results to a JSON file.

        **Example:**

        ```python
        {
            "start_time": "2024-02-28 10:00:00",
            "end_time": "2024-02-28 10:01:05",
            "duration": 65.0,
            "metrics": {
                "val": [
                    {
                        "val/AverageLoss": 0.0,
                        "val/BinaryAccuracy": 1.0,
                        "val/BinaryBalancedAccuracy": 1.0,
                        "val/BinaryPrecision": 1.0,
                        "val/BinaryRecall": 1.0,
                        "val/BinaryF1Score": 1.0,
                        "val/BinaryAUROC": 1.0
                    }
                ],
                "test": [
                    {
                        "test/AverageLoss": 0,0,
                        "test/BinaryAccuracy": 1.0,
                        "test/BinaryBalancedAccuracy": 1.0,
                        "test/BinaryPrecision": 1.0,
                        "test/BinaryRecall": 1.0,
                        "test/BinaryF1Score": 1.0,
                        "test/BinaryAUROC": 1.0
                    }
                ]
            }
        }
        ```

    Args:
        evaluation_results: Dictionary containing the evaluation results.
        results_path: Directory where the results should be saved.
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
