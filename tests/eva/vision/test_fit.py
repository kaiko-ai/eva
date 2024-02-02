"""Tests regarding the MLF fit command."""

import os

import pytest

from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/vision/tests/patch_camelyon.yaml",
    ],
)
def test_fit_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `fit` command with a given configuration relative path."""
    _cli.run_cli_from_main(
        cli_args=[
            "fit",
            "--config",
            os.path.join(lib_path, configuration_file),
            "--trainer.fast_dev_run",
            "True",
        ]
    )
