"""Tests regarding eva's CLI commands on vision datasets."""

import os
import tempfile
from unittest import mock

import pytest

from eva.vision.data import datasets
from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/dino_vit/online/bach.yaml",
        "configs/vision/dino_vit/online/crc.yaml",
        "configs/vision/dino_vit/online/mhist.yaml",
        "configs/vision/dino_vit/online/patch_camelyon.yaml",
        "configs/vision/dino_vit/offline/bach.yaml",
        "configs/vision/dino_vit/offline/crc.yaml",
        "configs/vision/dino_vit/offline/mhist.yaml",
        "configs/vision/dino_vit/offline/patch_camelyon.yaml",
        "configs/vision/owkin/phikon/offline/bach.yaml",
        "configs/vision/owkin/phikon/offline/crc.yaml",
        "configs/vision/owkin/phikon/offline/mhist.yaml",
        "configs/vision/owkin/phikon/offline/patch_camelyon.yaml",
    ],
)
def test_configuration_initialization(configuration_file: str, lib_path: str) -> None:
    """Tests that a given configuration file can be initialized."""
    _cli.run_cli_from_main(
        cli_args=[
            "fit",
            "--config",
            os.path.join(lib_path, configuration_file),
            "--print_config",
        ]
    )


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/tests/online/patch_camelyon.yaml",
    ],
)
def test_fit_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `fit` command with a given configuration file."""
    _skip_dataset_validation()
    _cli.run_cli_from_main(
        cli_args=[
            "fit",
            "--config",
            os.path.join(lib_path, configuration_file),
        ]
    )


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/tests/offline/patch_camelyon.yaml",
    ],
)
def test_predict_fit_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `predict_fit` command with a given configuration file."""
    _skip_dataset_validation()
    with tempfile.TemporaryDirectory() as output_dir:
        with mock.patch.dict(os.environ, {"EMBEDDINGS_ROOT": output_dir}):
            _cli.run_cli_from_main(
                cli_args=[
                    "predict_fit",
                    "--config",
                    os.path.join(lib_path, configuration_file),
                ]
            )


def _skip_dataset_validation() -> None:
    """Mocks the validation step of the datasets."""
    datasets.PatchCamelyon.validate = mock.MagicMock(return_value=None)
