"""Tests regarding eva's CLI commands on vision datasets."""

import os
import tempfile
from unittest import mock
from unittest.mock import patch

import pytest

from eva.vision.data import datasets
from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        # | online
        # classification
        "configs/vision/pathology/online/classification/bach.yaml",
        "configs/vision/pathology/online/classification/bracs.yaml",
        "configs/vision/pathology/online/classification/breakhis.yaml",
        "configs/vision/pathology/online/classification/crc.yaml",
        "configs/vision/pathology/online/classification/gleason_arvaniti.yaml",
        "configs/vision/pathology/online/classification/mhist.yaml",
        "configs/vision/pathology/online/classification/patch_camelyon.yaml",
        "configs/vision/pathology/online/classification/unitopatho.yaml",
        # segmentation
        "configs/vision/pathology/online/segmentation/bcss.yaml",
        "configs/vision/pathology/online/segmentation/consep.yaml",
        "configs/vision/pathology/online/segmentation/monusac.yaml",
        "configs/vision/radiology/online/segmentation/lits17.yaml",
        "configs/vision/radiology/online/segmentation/msd_task7_pancreas.yaml",
        # | offline
        # classification
        "configs/vision/pathology/offline/classification/bach.yaml",
        "configs/vision/pathology/offline/classification/bracs.yaml",
        "configs/vision/pathology/offline/classification/breakhis.yaml",
        "configs/vision/pathology/offline/classification/camelyon16.yaml",
        "configs/vision/pathology/offline/classification/crc.yaml",
        "configs/vision/pathology/offline/classification/gleason_arvaniti.yaml",
        "configs/vision/pathology/offline/classification/mhist.yaml",
        "configs/vision/pathology/offline/classification/panda.yaml",
        "configs/vision/pathology/offline/classification/patch_camelyon.yaml",
        "configs/vision/pathology/offline/classification/unitopatho.yaml",
        # segmentation
        "configs/vision/pathology/offline/segmentation/bcss.yaml",
        "configs/vision/pathology/offline/segmentation/consep.yaml",
        "configs/vision/pathology/offline/segmentation/monusac.yaml",
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
        "configs/vision/tests/online/consep.yaml",
        "configs/vision/tests/online/lits17.yaml",
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
        "configs/vision/tests/offline/panda.yaml",
        "configs/vision/tests/offline/consep.yaml",
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
    datasets.CoNSeP.validate = mock.MagicMock(return_value=None)
    datasets.LiTS17.validate = mock.MagicMock(return_value=None)
    datasets.LiTS17._split_index_ranges = {  # type: ignore
        "train": [(31, 32)],
        "val": [(45, 46)],
    }


@pytest.fixture(autouse=True)
def mock_download():
    """Mocks the download functions to avoid downloading resources when running tests."""
    with patch.object(datasets.PANDA, "_download_resources", return_value=None):
        yield
