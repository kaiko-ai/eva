"""Main interface class."""

from torch import nn

from eva.core import trainers as eva_trainer
from eva.core.data import datamodules
from eva.core.models import modules


class Interface:
    """A high-level interface for training and validating a machine learning model.

    This class provides a convenient interface to connect a model, data, and trainer
    to train and validate a model.
    """

    def fit(
        self,
        backbone: nn.Module,
        trainer: eva_trainer.Trainer,
        model: modules.ModelModule,
        data: datamodules.DataModule,
    ) -> None:
        """Perform model training and evaluation out-of-place.

        This method uses the specified trainer to fit the model using the provided data.

        Example use cases:

        - Using a model consisting of a frozen backbone and a head, the backbone will generate
          the embeddings on the fly which are then used as input features to train the head on
          the downstream task specified by the given dataset.
        - Fitting only the head network using a dataset that loads pre-computed embeddings.

        Args:
            backbone: The backbone model to use.
            trainer: The base trainer to use.
            model: The model module to use.
            data: The data module.
        """
        trainer.run_evaluation_session(model=model, datamodule=data)

    def predict(
        self,
        backbone: nn.Module,
        trainer: eva_trainer.Trainer,
        model: modules.ModelModule,
        data: datamodules.DataModule,
    ) -> None:
        """Perform model prediction out-of-place.

        This method performs inference with a pre-trained foundation model to compute embeddings.

        Args:
            backbone: The backbone model to use.
            trainer: The base trainer to use.
            model: The model module to use.
            data: The data module.
        """
        eva_trainer.infer_model(
            base_trainer=trainer,
            base_model=model,
            datamodule=data,
            return_predictions=False,
        )

    def predict_fit(
        self,
        backbone: nn.Module,
        trainer: eva_trainer.Trainer,
        model: modules.ModelModule,
        data: datamodules.DataModule,
    ) -> None:
        """Combines the predict and fit commands in one method.

        This method performs the following two steps:
        1. predict: perform inference with a pre-trained foundation model to compute embeddings.
        2. fit: training the head network using the embeddings generated in step 1.

        Args:
            backbone: The backbone model to use.
            trainer: The base trainer to use.
            model: The model module to use.
            data: The data module.
        """
        self.predict(backbone=backbone, trainer=trainer, model=model, data=data)
        self.fit(backbone=backbone, trainer=trainer, model=model, data=data)
