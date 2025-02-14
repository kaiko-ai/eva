# type: ignore
"""Utility functions for logging with Weights & Biases."""

from typing import Any, Dict

from loguru import logger


def rename_active_run(name: str) -> None:
    """Renames the current run."""
    import wandb

    if wandb.run:
        wandb.run.name = name
        wandb.run.save()
    else:
        logger.warning("No active wandb run found that could be renamed.")


def init_run(name: str, init_kwargs: Dict[str, Any]) -> None:
    """Initializes a new run. If there is an active run, it will be renamed and reused."""
    import wandb

    init_kwargs["name"] = name
    rename_active_run(name)
    wandb.init(**init_kwargs)


def finish_run() -> None:
    """Finish the current run."""
    import wandb

    wandb.finish()
