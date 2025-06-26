"""Tests regarding eva's CLI commands on language datasets."""

import os
from unittest import mock
from unittest.mock import patch

import pytest
from datasets import Dataset

from eva.language.data import datasets
from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/language/pubmedqa.yaml",
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
        "configs/language/pubmedqa.yaml",
    ],
)
def test_validate_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `validate` command with a given configuration file."""
    with mock.patch.dict(os.environ, {"N_RUNS": "1", "BATCH_SIZE": "2"}):
        _cli.run_cli_from_main(
            cli_args=[
                "validate",
                "--config",
                os.path.join(lib_path, configuration_file),
            ]
        )


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mocks external dependencies to avoid API calls and downloads."""

    def _fake_completion(_model, _messages, **_kwargs):
        return {"choices": [{"message": {"content": "yes"}}]}

    def _fake_prepare_data(self):
        # Create a minimal fake dataset matching PubMedQA format
        self.dataset = Dataset.from_dict(
            {
                "QUESTION": ["Test question?"],
                "CONTEXTS": [["Test context"]],
                "final_decision": ["yes"],
            }
        )

    with (
        patch.object(datasets.PubMedQA, "prepare_data", _fake_prepare_data),
        patch(
            "eva.language.models.wrappers.litellm.batch_completion",
            lambda **_kwargs: [_fake_completion(None, None)],
        ),
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}),
    ):
        yield
