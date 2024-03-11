"""Multi-run summary recorder."""

import collections
import json
import os
import statistics
from typing import Any, Dict, List, Mapping

from lightning_fabric.utilities import cloud_io
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT
from toolz import dicttoolz

SESSION_METRICS = Mapping[str, List[float]]
"""Session metrics type-hint."""


class SessionRecorder:
    """Multi-run (session) summary logger."""

    def __init__(
        self,
        output_dir: str,
        save_as: str = "results.json",
    ) -> None:
        """Initializes the recorder.

        Args:
            output_dir: The destination folder to save the results.
            save_as: The output filename.
        """
        self._output_dir = output_dir
        self._save_as = save_as

        self._validation_metrics: List[SESSION_METRICS] = []
        self._test_metrics: List[SESSION_METRICS] = []

    @property
    def filename(self) -> str:
        """Returns the output filename."""
        return os.path.join(self._output_dir, self._save_as)

    def update(
        self,
        validation_scores: _EVALUATE_OUTPUT,
        test_scores: _EVALUATE_OUTPUT | None = None,
    ) -> None:
        """Updates the state of the tracked metrics in-place."""
        self._update_validation_metrics(validation_scores)
        self._update_test_metrics(test_scores)

    def compute(self) -> Dict[str, List[Dict[str, Any]]]:
        """Computes and returns the session statistics."""
        validation_statistics = list(map(_calculate_statistics, self._validation_metrics))
        test_statistics = list(map(_calculate_statistics, self._test_metrics))
        return {"val": validation_statistics, "test": test_statistics}

    def export(self) -> Dict[str, Any]:
        """Exports the results."""
        statistics = self.compute()
        return {"metrics": statistics}

    def save(self) -> None:
        """Saves the recorded results."""
        results = self.export()
        _save_json(results, self.filename)

    def reset(self) -> None:
        """Resets the state of the tracked metrics."""
        self._validation_metrics = []
        self._test_metrics = []

    def _update_validation_metrics(self, metrics: _EVALUATE_OUTPUT) -> None:
        """Updates the validation metrics in-place."""
        self._validation_metrics = _update_session_metrics(self._validation_metrics, metrics)

    def _update_test_metrics(self, metrics: _EVALUATE_OUTPUT | None) -> None:
        """Updates the test metrics in-place."""
        if metrics:
            self._test_metrics = _update_session_metrics(self._test_metrics, metrics)


def _update_session_metrics(
    session_metrics: List[SESSION_METRICS],
    run_metrics: _EVALUATE_OUTPUT,
) -> List[SESSION_METRICS]:
    """Updates and returns the given metrics session with the new ones."""
    session_metrics = session_metrics or _init_session_metrics(len(run_metrics))
    for index, dataset_metrics in enumerate(run_metrics):
        for name, value in dataset_metrics.items():
            session_metrics[index][name].append(value)
    return session_metrics


def _init_session_metrics(n_datasets: int) -> List[SESSION_METRICS]:
    """Returns the init session metrics."""
    return [collections.defaultdict(list) for _ in range(n_datasets)]


def _calculate_statistics(session_metrics: SESSION_METRICS) -> Dict[str, float | List[float]]:
    """Calculate the metric statistics of a dataset session run."""

    def _calculate_metric_statistics(values: List[float]) -> Dict[str, float | List[float]]:
        """Calculates and returns the metric statistics."""
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        return {"mean": mean, "stdev": stdev, "values": values}

    return dicttoolz.valmap(_calculate_metric_statistics, session_metrics)


def _save_json(data: Dict[str, Any], save_as: str = "data.json"):
    """Saves data to a json file."""
    if not save_as.endswith(".json"):
        raise ValueError()

    output_dir = os.path.dirname(save_as)
    fs = cloud_io.get_filesystem(output_dir, anon=False)
    fs.makedirs(output_dir, exist_ok=True)
    with fs.open(save_as, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)