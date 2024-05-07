"""JSON based experimental logger."""

import json
import os
from argparse import Namespace
from typing import Any, Dict, List, Union

import torch
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities import cloud_io, rank_zero
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.logger import Logger
from loguru import logger
from toolz import dicttoolz
from typing_extensions import override


class JSONLogger(Logger):
    """Local file system experimental logger in JSON format.

    Logs are saved to ``os.path.join(root_dir, name, version)``.
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        name: str | None = None,
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

        self._fs = cloud_io.get_filesystem(self._root_dir, anon=False)
        self._experiment: _ExperimentWriter | None = None

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Returns the experiment writer object."""
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @property
    @override
    def root_dir(self) -> str | _PATH:
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
        self, metrics: Dict[str, torch.Tensor | float], step: int | None = None
    ) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)  # type: ignore
        self.experiment.log_metrics(metrics, step=step)
        if (step or 0 + 1) % self._flush_logs_every_n_steps == 0:
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
            logger.warning("Missing logger folder: %s", versions_root)
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if cloud_io._is_dir(self._fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class _ExperimentWriter:
    """Experiment writer for JSONLogger."""

    NAME_METRICS_FILE = "metrics.json"
    ENCODING = "utf-8"

    def __init__(self, log_dir: str) -> None:
        """Initializes the experimental writer.

        Args:
            log_dir: Directory for the experiment logs
        """
        self.log_dir = log_dir

        self.metrics: List[Dict[str, float]] = []
        self._fs = cloud_io.get_filesystem(log_dir, anon=False)

        self._check_log_dir_exists()
        self._fs.makedirs(self.log_dir, exist_ok=True)

    @property
    def filename(self) -> str:
        return os.path.join(self.log_dir, self.NAME_METRICS_FILE)

    def log_metrics(
        self, metrics_dict: Dict[str, torch.Tensor | float], step: int | None = None
    ) -> None:
        """Record metrics."""

        def _handle_value(value: Union[torch.Tensor, Any]) -> Any:
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        item = dicttoolz.valmap(_handle_value, metrics_dict)
        item["step"] = step
        self.metrics.append(item)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        self._update_metrics()
        self._save_json()
        self._reset_metrics()

    def _check_log_dir_exists(self) -> None:
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero.rank_zero_warn(  # type: ignore
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new"
                " ones are saved!"
            )
            if self._fs.isfile(self.filename):
                self._fs.rm_file(self.filename)

    def _save_json(
        self,
        *,
        indent: int | None = 2,
    ) -> None:
        """Saves data to a JSON file.

        Args:
            indent: The number of spaces per level that should be used to indent
                the content. An indent level of 0 or negative will only insert
                newlines. `None` selects the most compact representation.
        """
        with self._fs.open(self.filename, mode="w", encoding=self.ENCODING) as file:
            json.dump(self.metrics, file, indent=indent)

    def _load_json(self) -> Any:
        """Loads a JSON file and returns its raw data."""
        with self._fs.open(self.filename, "r", encoding=self.ENCODING) as file:
            return json.load(file)

    def _update_metrics(self) -> None:
        """Appends the new tracked metrics with the old ones."""
        file_exists = self._fs.isfile(self.filename)
        if file_exists:
            previous_metrics = self._load_json()
            self.metrics = previous_metrics + self.metrics

    def _reset_metrics(self) -> None:
        """Resets metrics."""
        self.metrics = []
