"""Tests regarding the EVA `fit` CLI command on core datasets."""

import os

import pytest

from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/core/tests/offline/embeddings.yaml",
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
