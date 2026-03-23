"""Tests the HeadModule module."""

from unittest import mock

import pytest
import torch
from monai.inferers.inferer import SimpleInferer
from torch import nn

from eva.core import metrics, trainers
from eva.core.data import datamodules, datasets
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


def test_semantic_segmentation_module_forward_with_inferer(
    model_with_inferer: modules.SemanticSegmentationModule,
    segmentation_dataset: datasets.TorchDataset,
) -> None:
    """Tests if the forward pass uses the inferer when in eval mode."""
    # Get a sample from dataset
    sample_data, _ = segmentation_dataset[0]
    sample_data = sample_data.unsqueeze(0)
    to_size = sample_data.shape[2:]

    # Execute the forward pass in train & eval mode
    assert model_with_inferer.inferer is not None, "Inferer should not be None"
    with mock.patch.object(
        type(model_with_inferer.inferer), "__call__", wraps=model_with_inferer.inferer.__call__
    ) as mock_inferer_call:
        output = model_with_inferer(sample_data, to_size=to_size)
        mock_inferer_call.assert_not_called()

        model_with_inferer.eval()
        output = model_with_inferer(sample_data, to_size=to_size)
        mock_inferer_call.assert_called_once()

    assert output.shape[0] == 1
    assert output.shape[1] == 4
    assert output.shape[2:] == (16, 16)


model_kwargs = {
    "decoder": segmentation.Decoder2D(
        layers=nn.Conv2d(
            in_channels=192,
            out_channels=4,
            kernel_size=(1, 1),
        ),
    ),
    "criterion": nn.CrossEntropyLoss(),
    "encoder": wrappers.TimmModel(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        out_indices=1,
        model_kwargs={
            "dynamic_img_size": True,
        },
    ),
    "metrics": metrics.MetricsSchema(
        common=metrics.AverageLoss(),
    ),
}


@pytest.fixture(scope="function")
def model() -> modules.SemanticSegmentationModule:
    """Returns a SemanticSegmentationModule model fixture."""
    return modules.SemanticSegmentationModule(**model_kwargs)


@pytest.fixture(scope="function")
def model_with_inferer(
    model: modules.SemanticSegmentationModule,
) -> modules.SemanticSegmentationModule:
    """Returns a SemanticSegmentationModule model fixture with an inferer."""
    return modules.SemanticSegmentationModule(**model_kwargs, inferer=SimpleInferer())
