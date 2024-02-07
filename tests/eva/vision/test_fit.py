"""Tests regarding the EVA `fit` CLI command on vision datasets."""

import os

import pytest

from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/tests/patch_camelyon.yaml",
    ],
)
def test_fit_online(configuration_file: str, lib_path: str):
    """Tests CLI `fit` command with a given configuration."""
    _cli.run_cli_from_main(cli_args=["fit", "--config", os.path.join(lib_path, configuration_file)])


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/tests/precalculated_embeddings/patches.yaml",
        "configs/vision/tests/precalculated_embeddings/slides.yaml",
    ],
)
def test_fit_from_precalculated_embeddings(configuration_file: str, lib_path: str):
    """Tests CLI `fit` command with a given configuration using precalculated embeddings."""
    _cli.run_cli_from_main(cli_args=["fit", "--config", os.path.join(lib_path, configuration_file)])
