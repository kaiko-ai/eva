"""Tests the HeadModule module."""

import pytest
import torch
from torch import nn

from eva.core import metrics, trainers
from eva.core.data import datamodules
from eva.vision.models import modules, wrappers
from eva.vision.models.networks.decoders import segmentation


@pytest.mark.parametrize(
    "dataset_fixture",
    [
        "segmentation_dataset",
        "segmentation_dataset_with_metadata",
    ],
)
def test_semantic_segmentation_module_fit(
    model: modules.SemanticSegmentationModule,
    datamodule: datamodules.DataModule,
    trainer: trainers.Trainer,
) -> None:
    """Tests the SemanticSegmentationModule fit pipeline."""
    initial_decoder_weights = model.decoder._layers.weight.clone()
    trainer.fit(model, datamodule=datamodule)
    # verify that the metrics were updated
    assert trainer.logged_metrics["train/AverageLoss"] > 0
    assert trainer.logged_metrics["val/AverageLoss"] > 0
    # verify that head weights were updated
    assert not torch.all(torch.eq(initial_decoder_weights, model.decoder._layers.weight))


@pytest.fixture(scope="function")
def model(n_classes: int = 4) -> modules.SemanticSegmentationModule:
    """Returns a SemanticSegmentationModule model fixture."""
    return modules.SemanticSegmentationModule(
        decoder=segmentation.Decoder2D(
            layers=nn.Conv2d(
                in_channels=192,
                out_channels=n_classes,
                kernel_size=(1, 1),
            ),
        ),
        criterion=nn.CrossEntropyLoss(),
        encoder=wrappers.TimmModel(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            out_indices=1,
            model_kwargs={
                "dynamic_img_size": True,
            },
        ),
        metrics=metrics.MetricsSchema(
            common=metrics.AverageLoss(),
        ),
    )
