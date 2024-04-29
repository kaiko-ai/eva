"""JSON based experimental logger."""

import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union
from toolz import dicttoolz
from lightning.fabric.loggers import logger
from lightning.fabric.utilities import cloud_io, rank_zero
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.types import _PATH
from torch import Tensor
from typing_extensions import override

log = logging.getLogger(__name__)


class JSONLogger(logger.Logger):
    """Local file system experimental logger in JSON format.

    Logs are saved to ``os.path.join(root_dir, name, version)``.
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        name: str | None = "logs",
        version: int | str | None = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        """Initializes the logger.

        Args:
            root_dir: The root directory in which all the experiments with
                different names and versions will be stored.
            name: The name of the experiment name. If name is `None`, logs
                (versions) will be stored to the save dir directly.
            version: The experiment version. If version is not specified the
                logger inspects the save directory for existing versions, then
                automatically assigns the next available version. If the version
                is specified, and the directory already contains a metrics file
                for that version, it will be overwritten.
            prefix: A string to put at the beginning of metric keys.
            flush_logs_every_n_steps: How often to flush logs to disk.
        """
        super().__init__()

        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

        self._fs = cloud_io.get_filesystem(self._root_dir)
        self._experiment: _JSONExperimentWriter

    @property
    @logger.rank_zero_experiment
    def experiment(self) -> "_JSONExperimentWriter":
        """Returns the experiment writer object."""
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _JSONExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @property
    @override
    def root_dir(self) -> str:
        return self._root_dir

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self) -> int | str:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)

    @override
    @rank_zero.rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any] | Namespace) -> None:
        raise NotImplementedError("The `JSONLogger` does not support logging hyperparameters.")

    @override
    @rank_zero.rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Tensor | float], step: int | None = None
    ) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        # if step is None:
        #     step = len(self.experiment.metrics)
        self.experiment.log_metrics(metrics, step)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @override
    @rank_zero.rank_zero_only
    def save(self) -> None:
        self.experiment.save()

    @override
    @rank_zero.rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on
            # the main process, as no experiment has been initialized there.
            return
        self.save()

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)
        if not cloud_io._is_dir(self._fs, versions_root, strict=True):
            log.warning(f"Missing logger folder '{versions_root}'.")
            return 0

        latest_version = -1
        for directory in self._fs.listdir(versions_root):
            full_path, name = directory["name"], os.path.basename(directory["name"])
            if cloud_io._is_dir(self._fs, full_path) and name.startswith("version_"):
                version = name.split("_")[-1]
                if version.isdigit():
                    latest_version = max(latest_version, version)

        return max(latest_version, 0)

        existing_versions = []
        for directory in self._fs.listdir(versions_root):
            full_path, name = directory["name"], os.path.basename(directory["name"])
            if cloud_io._is_dir(self._fs, full_path) and name.startswith("version_"):
                version = name.split("_")[1]
                if version.isdigit():
                    existing_versions.append(int(version))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class _JSONExperimentWriter:
    """Experiment writer for JSONLogger."""

    NAME_METRICS_FILE = "metrics.json"

    def __init__(self, log_dir: str) -> None:
        """Initializes the experimental writer.
        
        Args:
            log_dir: Directory for the experiment logs
        """
        self.log_dir = log_dir

        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        self._fs = cloud_io.get_filesystem(log_dir)
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

        self._check_log_dir_exists()
        self._fs.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(self, metrics_dict: Dict[str, float], step: int | None = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        # metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics = dicttoolz.valmap(_handle_value, metrics_dict)
        metrics["step"] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(
            self.metrics_file_path, mode=("a" if file_exists else "w"), newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _check_log_dir_exists(self) -> None:
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero.rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
            if self._fs.isfile(self.metrics_file_path):
                self._fs.rm_file(self.metrics_file_path)
