"""YAML related I/O operations."""

import functools
from pathlib import Path
from typing import Any

import omegaconf


@functools.singledispatch
def save_yaml(data: Any, save_path: str) -> None:
    """Save data to a YAML file.

    Args:
        data: Data to be saved
        save_path: Path to save the data
    """
    raise TypeError(f"Unsupported type: {type(data)}.")


@save_yaml.register
def _(
    data: omegaconf.DictConfig | omegaconf.ListConfig, save_path: str, resolve: bool = False
) -> None:
    """Save OmegaConf DictConfig to a YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as file:
        data_yaml = omegaconf.OmegaConf.to_yaml(data, resolve=resolve)
        file.write(data_yaml)


def update_keys(
    yaml_path_a: str, yaml_path_b: str, merge: bool = False, resolve: bool = False
) -> str:
    """Updates the keys of a YAML file with the values from another YAML file.

    Args:
        yaml_path_a: Path to the base YAML file
        yaml_path_b: Path to the YAML file with the keys to be updated
        merge: Whether to merge the values of the keys or replace them
        resolve: Whether to resolve the references in the YAML file

    Return:
        str: Updated YAML as string
    """
    config = omegaconf.OmegaConf.load(yaml_path_a)
    override = omegaconf.OmegaConf.load(yaml_path_b)

    if not isinstance(config, omegaconf.DictConfig) or not isinstance(
        override, omegaconf.DictConfig
    ):
        raise TypeError("Only yaml files in dict format are supported.")

    for key in override.keys():
        omegaconf.OmegaConf.update(config, key, override[key], merge=merge)  # type: ignore

    return omegaconf.OmegaConf.to_yaml(config, resolve=resolve)
