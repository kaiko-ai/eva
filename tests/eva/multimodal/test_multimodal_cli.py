"""Tests regarding eva's CLI commands on multimodal datasets."""

import os
import tempfile
from unittest import mock
from unittest.mock import patch

import pytest

from eva.multimodal.data import datasets
from tests.eva import _cli

BATCH_SIZE = 2


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml",
    ],
)
def test_configuration_initialization(configuration_file: str, lib_path: str) -> None:
    """Tests that a given configuration file can be initialized."""
    _cli.run_cli_from_main(
        cli_args=[
            "validate",
            "--config",
            os.path.join(lib_path, configuration_file),
            "--print_config",
        ]
    )


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml",
    ],
)
def test_validate_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `validate` command with a given configuration file."""
    with mock.patch.dict(os.environ, {"N_RUNS": "1", "BATCH_SIZE": f"{BATCH_SIZE}"}):
        _cli.run_cli_from_main(
            cli_args=[
                "validate",
                "--config",
                os.path.join(lib_path, configuration_file),
            ]
        )


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml",
    ],
)
def test_test_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `test` command with a given configuration file."""
    with mock.patch.dict(os.environ, {"N_RUNS": "1", "BATCH_SIZE": f"{BATCH_SIZE}"}):
        _cli.run_cli_from_main(
            cli_args=[
                "test",
                "--config",
                os.path.join(lib_path, configuration_file),
            ]
        )


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/multimodal/pathology/offline/multiple_choice/patch_camelyon.yaml",
    ],
)
def test_predict_validate_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `predict` and `validate` commands with a given configuration file."""
    with tempfile.TemporaryDirectory() as output_dir:
        with mock.patch.dict(
            os.environ,
            {"N_RUNS": "1", "PREDICT_BATCH_SIZE": "2", "PREDICTIONS_OUTPUT_DIR": output_dir},
        ):
            _cli.run_cli_from_main(
                cli_args=[
                    "predict",
                    "--config",
                    os.path.join(lib_path, configuration_file),
                ]
            )
            _cli.run_cli_from_main(
                cli_args=[
                    "validate",
                    "--config",
                    os.path.join(lib_path, configuration_file),
                ]
            )


@pytest.fixture(autouse=True)
def skip_dataset_validation() -> None:
    """Mocks the validation step of the datasets."""
    datasets.PatchCamelyon.validate = mock.MagicMock(return_value=None)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mocks external dependencies to avoid API calls and downloads."""

    def _fake_completion():
        return {"choices": [{"message": {"content": "A", "role": "assistant"}}]}

    with (
        patch(
            "eva.language.models.wrappers.litellm.batch_completion",
            lambda **_kwargs: [_fake_completion()] * BATCH_SIZE,
        ),
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}),
        mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy-key"}),
    ):
        yield
