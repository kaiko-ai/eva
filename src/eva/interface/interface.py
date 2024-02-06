"""Main interface class."""

from eva import trainers
from eva.data import datamodules
from eva.models import modules


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
        trainer.fit(model=model, datamodule=data)
        trainer.test(datamodule=data)
