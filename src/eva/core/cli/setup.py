"""Operations which are executed when the CLI is triggered."""

import os
import sys
import uuid
import warnings
from pathlib import Path

import jsonargparse
import yaml
from lightning_fabric.utilities import seed as pl_seed
from loguru import logger

import eva.core.utils.io.yaml as eva_yaml
from eva.core.utils import workers


def _configure_random_seed(seed: int | None = None) -> None:
    """Sets the global random seed.

    Args:
        seed: The seed number to use. If `None`, it will read the seed from
            `EVA_GLOBAL_SEED` env variable. If `None` and the `EVA_GLOBAL_SEED`
            env variable is not set, then the seed defaults to `42`. If `None`
            and the `EVA_GLOBAL_SEED` is set to `False`, it will not set the seed.
    """
    effective_seed = seed or os.environ.get("EVA_GLOBAL_SEED", default=42)
    if isinstance(effective_seed, str):
        effective_seed = yaml.safe_load(effective_seed)
        if not isinstance(effective_seed, (bool, int)):
            raise ValueError(
                f"Invalid 'EVA_GLOBAL_SEED' value '{effective_seed}'. "
                "It should be an integer or a boolean value."
            )

    if isinstance(effective_seed, bool) and effective_seed is False:
        return

    pl_seed.seed_everything(seed=int(effective_seed), workers=True)


def _configure_jsonargparse() -> None:
    """Configures the `jsonargparse` library."""
    jsonargparse.set_config_read_mode(
        urls_enabled=True,
        fsspec_enabled=True,
    )


def _initialize_logger() -> None:
    """Initializes, manipulates and customizes the logger.

    This customizable logger can be used by just importing `loguru`
    from everywhere as follows:
    >>> from loguru import logger
    >>> logger.info(...)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<blue>{time:HH:mm:ss}</blue>"
        " :: <bold><level>{level}</level></bold>"
        " :: {message}",
        colorize=True,
        level="INFO",
    )


def _suppress_warnings() -> None:
    """Suppress all warnings from all subprocesses."""
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"


def _enable_mps_fallback() -> None:
    """It enables the MPS fallback in torch.

    Note that this action has to take place before importing torch.
    """
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _merge_configs(save_dir: str = os.path.join(str(Path.home()), ".cache/eva/configs")) -> None:
    """When multiple config files are specified in sys.argv, merges them into one.

    Args:
        save_dir: Directory to save the merged config file.
    """
    config_paths, remaining_args = [], []
    for arg in sys.argv:
        if arg.strip().endswith(".yaml"):
            config_paths.append(arg)
        elif arg.strip() != "--config":
            remaining_args.append(arg)

    if len(config_paths) <= 1:
        return
    elif len(config_paths) == 2:
        config = eva_yaml.update_keys(config_paths[0], config_paths[1], merge=False, resolve=False)
        merged_path = os.path.join(save_dir, str(uuid.uuid4()), Path(config_paths[0]).name)
        Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w") as file:
            file.write(config)
        logger.info(f"Received multiple config files & merged them into {merged_path}.")
        sys.argv = remaining_args + ["--config", merged_path]
    else:
        raise ValueError("More than two --config files are not supported.")


@workers.main_worker_only
def setup() -> None:
    """Sets up the environment before the module is imported."""
    _configure_random_seed()
    _configure_jsonargparse()
    _initialize_logger()
    _suppress_warnings()
    _enable_mps_fallback()
    _merge_configs()
