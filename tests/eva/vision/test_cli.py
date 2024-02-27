"""Tests regarding the EVA `fit` CLI command on vision datasets."""

import os
import tempfile
from unittest.mock import patch

import pytest

from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/tests/patch_camelyon.yaml",
        "configs/vision/tests/offline/patches.yaml",
        "configs/vision/tests/offline/slides.yaml",
    ],
)
def test_fit_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `fit` command with a given configuration file."""
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
    with tempfile.TemporaryDirectory() as output_dir:
        with patch.dict(os.environ, {"EMBEDDINGS_ROOT": output_dir}):
            _cli.run_cli_from_main(
                cli_args=[
                    "predict_fit",
                    "--config",
                    os.path.join(lib_path, configuration_file),
                ]
            )
