"""Main interface class."""

from eva import trainers
from eva.data import datamodules
from eva.data.datamodules import schemas
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

        Example use cases:
        - Using a model consisting of a frozen backbone and a head, the backbone will generate
          the embeddings on the fly which are then used as input features to train the head on
          the downstream task specified by the given dataset.
        - Fitting only the head network using a dataset that loads pre-computed embeddings.

        Args:
            model: The model module.
            data: The data module.
            trainer: The trainer which processes the model and data.
        """
        trainers.fit_and_validate(trainer=trainer, model=model, datamodule=data)

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
