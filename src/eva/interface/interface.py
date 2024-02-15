"""Main interface class."""

import os
from datetime import datetime

from eva import trainers
from eva.data import datamodules
from eva.models import modules
from eva.utils.recorder import get_run_id, record_results


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
    ) -> None:
        """Perform model training and evaluation in place.

        This method uses the specified trainer to fit the model using the provided data.

        Args:
            model: The model module.
            data: The data module.
            trainer: The trainer which processes the model and data.
        """
        start_time = datetime.now()
        run_id = get_run_id(start_time)
        trainer.fit(model=model, datamodule=data)
        evaluation_results = {
            "val": trainer.validate(datamodule=data)[0],
            "test": trainer.test(datamodule=data)[0],
        }
        end_time = datetime.now()
        results_dir = os.path.join(trainer.default_root_dir, run_id)
        record_results(evaluation_results, results_dir, start_time, end_time)
