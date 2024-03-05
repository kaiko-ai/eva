"""Fit session related functions."""

import copy
from collections import abc
from typing import Any, List, Tuple

from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT

from eva.data import datamodules
from eva.models import modules
from eva.trainers import trainer as eva_trainer


def run_evaluation_session(
    base_trainer: eva_trainer.Trainer,
    base_model: modules.ModelModule,
    datamodule: datamodules.DataModule,
    *,
    n_runs: int = 1,
) -> List[Tuple[_EVALUATE_OUTPUT, _EVALUATE_OUTPUT | None]]:
    """Runs a downstream evaluation session.

    It fits and evaluates the model multiple times. Note
    that as the input `base_trainer` and `base_model` would
    be cloned, the input object would not be manipulated.

    Args:
        base_trainer: The base trainer module to use.
        base_model: The base model module to use.
        datamodule: The data module.
        n_runs: The amount of time to fit and evaluate the model.
    """
    scores = []
    for run_index in range(n_runs):
        trainer, model = _clone(base_trainer, base_model)
        trainer.setup_log_dirs(f"run_{run_index}")
        validation_scores, test_scores = fit_and_validate(trainer, model, datamodule)
        scores.append([validation_scores, test_scores])
    return scores


def fit_and_validate(
    trainer: eva_trainer.Trainer,
    model: modules.ModelModule,
    datamodule: datamodules.DataModule,
) -> Tuple[_EVALUATE_OUTPUT, _EVALUATE_OUTPUT | None]:
    """Fits and evaluates a model without altering the input objects.

    Note that, if set, it will evaluate the model on the test set as well.

    Args:
        trainer: The trainer module.
        model: The model module.
        datamodule: The data module.

    Returns:
        A tuple of with the validation and the test metrics (if exists).
    """
    trainer.fit(model, datamodule=datamodule)
    validation_scores = trainer.validate(datamodule=datamodule)
    test_scores = None if datamodule.datasets.test is None else trainer.test(datamodule=datamodule)
    return validation_scores, test_scores


def _clone(*inputs: Any) -> Any:
    """Deep copies a list of object and returns them."""
    if not isinstance(inputs, abc.Iterable):
        return copy.deepcopy(inputs)

    return [copy.deepcopy(obj) for obj in inputs]
