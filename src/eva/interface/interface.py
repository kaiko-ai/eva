"""Main interface class."""

import copy
import os
from datetime import datetime

import pytorch_lightning as pl
from loguru import logger

from eva import trainers
from eva.data import datamodules
from eva.data.datamodules import schemas
from eva.models import modules
from eva.utils.recorder import get_evaluation_id, record_results
from eva.vision.data import datasets


class Interface:
    """A high-level interface for training and validating a machine learning model.

    This class provides a convenient interface to connect a model, data, and trainer
    for training and validating the model using the specified data and trainer.
    """

    def fit(
        self,
        model: modules.ModelModule,
        data: datamodules.DataModule,
        trainer: trainers.Trainer,
        n_runs: int = 1,
    ) -> None:
        """Perform model training and evaluation in place.

        This method uses the specified trainer to fit the model using the provided data.

        Example usecases:
        - Using a model consisting of a frozen backbone and a head, the backbone will generate
          the embeddings on the fly which are then used as input features to train the head on
          the downstream task specified by the given dataset.
        - Fitting only the head network using a dataset that loads pre-computed embeddings.


        Args:
            model: The model module.
            data: The data module.
            trainer: The trainer which processes the model and data.
            n_runs: The number of runs to perform.
        """
        evaluation_id = get_evaluation_id()
        model = _adapt_model_module(model, data)

        for run_id in range(n_runs):
            trainer_run = copy.deepcopy(trainer)
            model_run = copy.deepcopy(model)
            log_dir = os.path.join(trainer_run.default_root_dir, evaluation_id, f"run_{run_id}")
            _adapt_log_dirs(trainer_run, log_dir)

            start_time = datetime.now()
            pl.seed_everything(run_id, workers=True)

            trainer_run.fit(model=model_run, datamodule=data)
            evaluation_results = {"val": trainer_run.validate(datamodule=data)}
            if data.datasets.test is not None:
                evaluation_results["test"] = trainer_run.test(datamodule=data)

            end_time = datetime.now()
            results_path = os.path.join(log_dir, "results.json")
            record_results(evaluation_results, results_path, start_time, end_time)

    def predict(
        self,
        model: modules.ModelModule,
        data: datamodules.DataModule,
        trainer: trainers.Trainer,
    ) -> None:
        """Perform model prediction in place.

        This method performs inference with a pre-trained foundation model to compute embeddings.

        Args:
            model: The model module.
            data: The data module.
            trainer: The trainer which processes the model and data.
        """
        predict_datamodule = datamodules.DataModule(
            dataloaders=schemas.DataloadersSchema(predict=data.dataloaders.predict),
            datasets=schemas.DatasetsSchema(predict=data.datasets.predict),
        )
        trainer.predict(model=model, datamodule=predict_datamodule, return_predictions=False)

    def predict_fit(
        self,
        model: modules.ModelModule,
        data: datamodules.DataModule,
        trainer: trainers.Trainer,
    ) -> None:
        """Combines the predict and fit commands in one method.

        This method performs the following two steps:
        1. predict: perform inference with a pre-trained foundation model to compute embeddings.
        2. fit: training the head network using the embeddings generated in step 1.

        Args:
            model: The model module.
            data: The data module.
            trainer: The trainer which processes the model and data.
        """
        self.predict(model=model, data=data, trainer=trainer)
        self.fit(model=model, data=data, trainer=trainer)


def _adapt_model_module(
    model: modules.ModelModule, data: datamodules.DataModule
) -> modules.ModelModule:
    """Adapts the model module based on the specified data module."""
    if isinstance(data.datasets.train, datasets.PatchEmbeddingDataset) and isinstance(
        model, modules.HeadModule
    ):
        model.backbone = None  # disable backbone when using pre-computed embeddings
    return model


def _adapt_log_dirs(trainer, log_dir) -> None:
    """Sets the log directory for the logger, trainer and callbacks."""
    for train_logger in trainer.loggers:
        try:
            train_logger.log_dir = log_dir
        except AttributeError:
            raise Warning(f"Could not set log_dir for logger {train_logger}")

    trainer.log_dir = log_dir
    if len(trainer.callbacks) > 0:
        model_checkpoint_callbacks = [
            c for c in trainer.callbacks if isinstance(c, pl.callbacks.ModelCheckpoint)
        ]
        if len(model_checkpoint_callbacks) > 0:
            model_checkpoint_callbacks[0].dirpath = log_dir
        else:
            logger.warning("No ModelCheckpoint callback found in trainer.callbacks")
