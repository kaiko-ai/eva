"""YAML related I/O operations."""

import omegaconf


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
