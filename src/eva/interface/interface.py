"""Main interface class."""
from eva import trainers
from eva.data import datamodules
from eva.models import modules


class Interface:
    """A high-level interface for training and validating a machine learning model.

    This class provides a convenient interface to connect a model, data, and trainer
    for training and validating the model using the specified data and trainer.
    """

    def __init__(
        self,
        model: modules.ModelModule,
        data: datamodules.DataModule,
        trainer: trainers.Trainer,
    ) -> None:
        """Initialize the Interface class.

        Args:
            model: The model module.
            data: The data module.
            trainer: The trainer which processes the model and data.
        """
        self._model = model
        self._data = data
        self._trainer = trainer

    def fit(self) -> None:
        """Perform model training in place.

        This method uses the specified trainer to fit the model using the provided data.
        """
        self._trainer.fit(self._model, datamodule=self._data)
