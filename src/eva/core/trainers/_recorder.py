"""Multi-run summary recorder."""

import collections
import json
import os
import statistics
import sys
from typing import Dict, List, Mapping, TypedDict

from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT
from lightning_fabric.utilities import cloud_io
from loguru import logger
from omegaconf import OmegaConf
from rich import console as rich_console
from rich import table as rich_table
from toolz import dicttoolz

SESSION_METRICS = Mapping[str, List[float]]
"""Session metrics type-hint."""


class SESSION_STATISTICS(TypedDict):
    """Type-hint for aggregated metrics of multiple runs with mean & stdev."""

    mean: float
    stdev: float
    values: List[float]


class STAGE_RESULTS(TypedDict):
    """Type-hint for metrics statstics for val & test stages."""

    val: List[Dict[str, SESSION_STATISTICS]]
    test: List[Dict[str, SESSION_STATISTICS]]


class RESULTS_DICT(TypedDict):
    """Type-hint for the final results dictionary."""

    metrics: STAGE_RESULTS


class SessionRecorder:
    """Multi-run (session) summary logger."""

    def __init__(
        self,
        output_dir: str,
        results_file: str = "results.json",
        config_file: str = "config.yaml",
        verbose: bool = True,
    ) -> None:
        """Initializes the recorder.

        Args:
            output_dir: The destination folder to save the results.
            results_file: The name of the results json file.
            config_file: The name of the yaml configuration file.
            verbose: Whether to print the session metrics.
        """
        self._output_dir = output_dir
        self._results_file = results_file
        self._config_file = config_file
        self._verbose = verbose

        self._validation_metrics: List[SESSION_METRICS] = []
        self._test_metrics: List[SESSION_METRICS] = []

    @property
    def filename(self) -> str:
        """Returns the output filename."""
        return os.path.join(self._output_dir, self._results_file)

    @property
    def config_path(self) -> str | None:
        """Returns the path to the .yaml configuration file from sys args if available."""
        if "--config" in sys.argv:
            try:
                config_path = sys.argv[sys.argv.index("--config") + 1]
                if not config_path.endswith(".yaml"):
                    logger.warning(f"Unexpected config file {config_path}, should be a .yaml file.")
                else:
                    return config_path
            except IndexError as e:
                logger.warning(f"Failed to fetch config_path from sys args {e}")

    def update(
        self,
        validation_scores: _EVALUATE_OUTPUT,
        test_scores: _EVALUATE_OUTPUT | None = None,
    ) -> None:
        """Updates the state of the tracked metrics in-place."""
        self._update_validation_metrics(validation_scores)
        self._update_test_metrics(test_scores)

    def compute(self) -> STAGE_RESULTS:
        """Computes and returns the session statistics."""
        validation_statistics = list(map(_calculate_statistics, self._validation_metrics))
        test_statistics = list(map(_calculate_statistics, self._test_metrics))
        return {"val": validation_statistics, "test": test_statistics}

    def export(self) -> RESULTS_DICT:
        """Exports the results."""
        statistics = self.compute()
        return {"metrics": statistics}

    def save(self) -> None:
        """Saves the recorded results."""
        results = self.export()
        _save_json(results, self.filename)
        self._save_config()
        if self._verbose:
            _print_results(results)

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

    def _save_config(self) -> None:
        """Saves the config yaml with resolved env placeholders to the output directory."""
        if self.config_path:
            config_fs = cloud_io.get_filesystem(self.config_path)
            with config_fs.open(self.config_path, "r") as config_file:
                config = OmegaConf.load(config_file)  # type: ignore

            fs = cloud_io.get_filesystem(self._output_dir, anon=False)
            with fs.open(os.path.join(self._output_dir, self._config_file), "w") as file:
                config_yaml = OmegaConf.to_yaml(config, resolve=True)
                file.write(config_yaml)


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


def _calculate_statistics(session_metrics: SESSION_METRICS) -> Dict[str, SESSION_STATISTICS]:
    """Calculate the metric statistics of a dataset session run."""

    def _calculate_metric_statistics(values: List[float]) -> SESSION_STATISTICS:
        """Calculates and returns the metric statistics."""
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        return {"mean": mean, "stdev": stdev, "values": values}

    return dicttoolz.valmap(_calculate_metric_statistics, session_metrics)


def _save_json(data: RESULTS_DICT, save_as: str = "data.json"):
    """Saves data to a json file."""
    if not save_as.endswith(".json"):
        raise ValueError()

    output_dir = os.path.dirname(save_as)
    fs = cloud_io.get_filesystem(output_dir, anon=False)
    fs.makedirs(output_dir, exist_ok=True)
    with fs.open(save_as, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _print_results(results: RESULTS_DICT) -> None:
    """Prints the results to the console."""
    try:
        for stage in ["val", "test"]:
            for dataset_idx in range(len(results["metrics"][stage])):
                _print_table(results["metrics"][stage][dataset_idx], stage, dataset_idx)
    except Exception as e:
        logger.error(f"Failed to print the results: {e}")


def _print_table(metrics_dict: Dict[str, SESSION_STATISTICS], stage: str, dataset_idx: int):
    """Prints the metrics of a single dataset as a table."""
    metrics_table = rich_table.Table(
        title=f"\n{stage.capitalize()} Dataset {dataset_idx}", title_style="bold"
    )
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Mean", style="magenta")
    metrics_table.add_column("Stdev", style="magenta")
    metrics_table.add_column("All", style="magenta")

    n_runs = len(metrics_dict[next(iter(metrics_dict))]["values"])
    for metric_name, metric_dict in metrics_dict.items():
        row = [
            metric_name,
            f'{metric_dict["mean"]:.3f}',
            f'{metric_dict["stdev"]:.3f}',
            ", ".join(f'{metric_dict["values"][i]:.3f}' for i in range(n_runs)),
        ]
        metrics_table.add_row(*row)

    console = rich_console.Console()
    console.print(metrics_table)
