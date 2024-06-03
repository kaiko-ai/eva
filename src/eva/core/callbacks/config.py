"""Configuration logger callback."""

import ast
import os
import sys
from types import BuiltinFunctionType
from typing import Any, Dict, List

import lightning.pytorch as pl
import yaml
from lightning_fabric.utilities import cloud_io
from loguru import logger as cli_logger
from omegaconf import OmegaConf
from typing_extensions import TypeGuard, override

from eva.core import loggers


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
        log_dir = trainer.log_dir
        if not _logdir_exists(log_dir):
            return

        configuration = _load_submitted_config()

        if self._verbose:
            config_as_text = yaml.dump(configuration, sort_keys=False)
            print(f"Configuration:\033[94m\n---\n{config_as_text}\033[0m")

        save_as = os.path.join(log_dir, self._save_as)
        fs = cloud_io.get_filesystem(log_dir)
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
    """Retrieves and loads the submitted configuration.

    Returns:
        The path to the configuration file.
    """
    config_paths = _fetch_submitted_config_path()
    return _load_yaml_files(config_paths)


def _fetch_submitted_config_path() -> List[str]:
    """Fetches the config path from command line arguments.

    Returns:
        The path to the configuration file.
    """
    return list(filter(lambda f: f.endswith(".yaml"), sys.argv))


def _load_yaml_files(paths: List[str]) -> Dict[str, Any]:
    """Loads yaml files and merge them from multiple paths.

    Args:
        paths: The paths to the yaml files.

    Returns:
        The merged configurations as a dictionary.
    """
    merged_config = {}
    for config_path in paths:
        fs = cloud_io.get_filesystem(config_path)
        with fs.open(config_path, "r") as file:
            omegaconf_file = OmegaConf.load(file)  # type: ignore
            config_dict = OmegaConf.to_object(omegaconf_file)  # type: ignore
            parsed_config = _type_resolver(config_dict)  # type: ignore
            merged_config.update(parsed_config)
    return merged_config


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
        elif isinstance(value, list) and isinstance(value[0], dict):
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
