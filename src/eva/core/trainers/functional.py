"""Fit session related functions."""

from typing import List, Literal, Tuple

from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT

from eva.core.data import datamodules
from eva.core.models import modules
from eva.core.trainers import _recorder, _utils
from eva.core.trainers import trainer as eva_trainer


def run_evaluation_session(
    base_trainer: eva_trainer.Trainer,
    base_model: modules.ModelModule,
    datamodule: datamodules.DataModule,
    *,
    n_runs: int = 1,
    stages: List[Literal["fit", "validate", "test"]] | None = None,
    verbose: bool = True,
) -> None:
    """Runs a downstream evaluation session out-of-place.

    It performs an evaluation run (with configurable stages) on the model
    multiple times. Note that as the input `base_trainer` and
    `base_model` would be cloned, the input object would not
    be modified.

    Args:
        base_trainer: The base trainer module to use.
        base_model: The base model module to use.
        datamodule: The data module.
        n_runs: The number of runs to perform.
        stages: List of stages to execute. Options: "fit", "validate", "test".
        verbose: Whether to verbose the session metrics instead of
            those of each individual run and vice-versa.
    """
    if not stages:
        stages = ["fit", "validate", "test"]
    recorder = _recorder.SessionRecorder(output_dir=base_trainer.default_log_dir, verbose=verbose)
    for run_index in range(n_runs):
        validation_scores, test_scores = run_evaluation(
            base_trainer,
            base_model,
            datamodule,
            run_id=run_index,
            stages=stages,
            verbose=not verbose,
        )
        if validation_scores:
            recorder.update(validation_scores, test_scores)
    recorder.save()


def run_evaluation(
    base_trainer: eva_trainer.Trainer,
    base_model: modules.ModelModule,
    datamodule: datamodules.DataModule,
    *,
    run_id: int | None = None,
    stages: List[Literal["fit", "validate", "test"]] | None = None,
    verbose: bool = True,
) -> Tuple[_EVALUATE_OUTPUT | None, _EVALUATE_OUTPUT | None]:
    """Runs the specified evaluation stages out-of-place.

    Args:
        base_trainer: The base trainer to use but not modify.
        base_model: The model module to use but not modify.
        datamodule: The data module.
        run_id: The run id to be appended to the output log directory.
            If `None`, it will use the log directory of the trainer as is.
        stages: List of stages to execute. Options: "fit", "validate", "test".
        verbose: Whether to print the validation and test metrics
            in the end of the training.

    Returns:
        A tuple with the validation and the test metrics (if executed).
        If a stage is not executed, its value will be None.
    """
    if not stages:
        stages = ["fit", "validate", "test"]
    trainer, model = _utils.clone(base_trainer, base_model)
    model.configure_model()

    trainer.init_logger_run(run_id)

    validation_scores = None
    test_scores = None

    if "fit" in stages:
        trainer.fit(model, datamodule=datamodule)
    if "validate" in stages:
        validation_scores = trainer.validate(
            model=model,
            datamodule=datamodule,
            verbose=verbose,
            ckpt_path=trainer.checkpoint_type,
        )
    if "test" in stages and getattr(datamodule.datasets, "test", None) is not None:
        test_scores = trainer.test(
            model=model,
            datamodule=datamodule,
            verbose=verbose,
            ckpt_path=trainer.checkpoint_type,
        )
    trainer.finish_logger_run(run_id)
    return validation_scores, test_scores


def infer_model(
    base_trainer: eva_trainer.Trainer,
    base_model: modules.ModelModule,
    datamodule: datamodules.DataModule,
    *,
    return_predictions: bool = False,
) -> None:
    """Performs model inference out-of-place.

    Note that the input `base_model` and `base_trainer` would
    not be modified.

    Args:
        base_trainer: The base trainer to use but not modify.
        base_model: The model module to use but not modify.
        datamodule: The data module.
        return_predictions: Whether to return the model predictions.
    """
    trainer, model = _utils.clone(base_trainer, base_model)
    return trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=return_predictions,
    )
