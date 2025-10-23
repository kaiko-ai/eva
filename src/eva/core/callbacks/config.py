"""Configuration logger callback."""

import ast
import os
import sys
from types import BuiltinFunctionType
from typing import Any, Dict, List

import lightning.pytorch as pl
import yaml
from lightning_fabric.utilities import cloud_io
from loguru import logger
from loguru import logger as cli_logger
from omegaconf import OmegaConf
from typing_extensions import TypeGuard, override

from eva.core import loggers
from eva.core.utils import distributed as dist_utils


class ConfigurationLogger(pl.Callback):
    """Logs the submitted configuration to the experimental logger."""

    _save_as: str = "config.yaml"

    def __init__(self, verbose: bool = True) -> None:
        """Initializes the callback.

        Args:
            verbose: Whether to print the configurations to print the
                configuration to the terminal.
        """
        super().__init__()

        self._verbose = verbose

    @override
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str | None = None,
    ) -> None:
        if dist_utils.is_distributed():
            logger.info("ConfigurationLogger skipped as not supported in distributed mode.")
            # TODO: Enabling leads to deadlocks in DDP mode, but I could not yet figure out why.
            return

        if not trainer.is_global_zero or not _logdir_exists(
            log_dir := trainer.log_dir, self._verbose
        ):
            return

        configuration = _load_submitted_config()

        if self._verbose:
            config_as_text = yaml.dump(configuration, sort_keys=False)
            print(f"Configuration:\033[94m\n---\n{config_as_text}\033[0m")

        save_as = os.path.join(log_dir, self._save_as)
        fs = cloud_io.get_filesystem(log_dir)

        if not fs.exists(log_dir):
            fs.makedirs(log_dir)

        with fs.open(save_as, "w") as output_file:
            yaml.dump(configuration, output_file, sort_keys=False)

        loggers.log_parameters(trainer.loggers, tag="configuration", parameters=configuration)


def _logdir_exists(logdir: str | None, verbose: bool = True) -> TypeGuard[str]:
    """Checks if the trainer has a log directory.

    Args:
        logdir: Trainer's logdir.
        name: The name to log with.
        verbose: Whether to log if it does not exist.

    Returns:
        A bool indicating if the log directory exists or not.
    """
    exists = isinstance(logdir, str)
    if not exists and verbose:
        print("\n")
        cli_logger.warning("Log directory is `None`. Configuration file will not be logged.\n")
    return exists


def _load_submitted_config() -> Dict[str, Any]:
    """Retrieves and loads the submitted configuration with CLI overrides.

    Returns:
        The resolved configuration as a dictionary.
    """
    config_paths = _fetch_submitted_config_path()
    cli_overrides = _parse_cli_overrides()
    return _load_yaml_files(config_paths, cli_overrides)


def _parse_cli_overrides() -> List[str]:
    """Parses CLI override arguments from sys.argv for trainer, model, and data.

    Supports the following formats:
    - Space-separated: --trainer.arg value
    - Equals-separated: --trainer.arg=value
    """
    argv = sys.argv[1:]
    valid_prefixes = ("--trainer.", "--model.", "--data.")
    overrides = []

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg.startswith(valid_prefixes):
            if "=" in arg:
                key, value = arg.split("=", 1)
                overrides.append(f"{key[2:]}={value}")
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                overrides.append(f"{arg[2:]}={argv[i + 1]}")
                i += 2
            else:
                i += 1
        else:
            i += 1

    return overrides


def _fetch_submitted_config_path() -> List[str]:
    """Fetches the config path from command line arguments.

    Returns:
        The path to the configuration file.
    """
    config_paths = list(filter(lambda f: f.endswith(".yaml"), sys.argv))
    return [p.replace("--config=", "") for p in config_paths]


def _load_yaml_files(paths: List[str], cli_overrides: List[str] | None = None) -> Dict[str, Any]:
    """Loads yaml files and merge them from multiple paths with CLI overrides.

    Args:
        paths: The paths to the yaml files.
        cli_overrides: Optional list of CLI overrides in dotlist format
            (e.g., ["trainer.n_runs=3"]).

    Returns:
        The merged configurations as a dictionary.
    """
    merged_config = OmegaConf.create({})

    for config_path in paths:
        fs = cloud_io.get_filesystem(config_path)
        with fs.open(config_path, "r") as file:
            omegaconf_file = OmegaConf.load(file)  # type: ignore
            merged_config = OmegaConf.merge(merged_config, omegaconf_file)

    if cli_overrides:
        cli_config = OmegaConf.from_dotlist(cli_overrides)
        merged_config = OmegaConf.merge(merged_config, cli_config)

    config_dict = OmegaConf.to_object(merged_config)  # type: ignore
    return _type_resolver(config_dict)  # type: ignore


def _type_resolver(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Parses the string values of a dictionary in-place.

    Args:
        mapping: A dictionary object.

    Returns:
        The mapping with the formatted values.
    """
    for key, value in mapping.items():
        if isinstance(value, dict):
            formatted_value = _type_resolver(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            formatted_value = [_type_resolver(subvalue) for subvalue in value]
        else:
            try:
                parsed_value = ast.literal_eval(value)  # type: ignore
                formatted_value = (
                    value if isinstance(parsed_value, BuiltinFunctionType) else parsed_value
                )
            except Exception:
                formatted_value = value
        mapping[key] = formatted_value
    return mapping
