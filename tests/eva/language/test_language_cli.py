"""Tests regarding eva's CLI commands on language datasets."""

import json
import os
import statistics
import tempfile
from typing import Any, Dict, List
from unittest import mock
from unittest.mock import patch

import pytest
from datasets import Dataset

from eva.language.data import datasets
from tests.eva import _cli


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/language/pathology/online/multiple_choice/pubmedqa.yaml",
        "configs/language/pathology/offline/multiple_choice/pubmedqa.yaml",
        "configs/language/tests/offline/pubmedqa_combine.yaml",
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
        "configs/language/pathology/online/multiple_choice/pubmedqa.yaml",
    ],
)
def test_validate_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `validate` command with a given configuration file."""
    with mock.patch.dict(os.environ, {"N_RUNS": "1", "BATCH_SIZE": "2", "MISSING_LIMIT": "0"}):
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
        "configs/language/pathology/offline/multiple_choice/pubmedqa.yaml",
    ],
)
def test_predict_validate_from_configuration(configuration_file: str, lib_path: str) -> None:
    """Tests CLI `predict` and `validate` commands with a given configuration file."""
    with tempfile.TemporaryDirectory() as output_dir:
        with mock.patch.dict(
            os.environ,
            {
                "N_RUNS": "1",
                "BATCH_SIZE": "2",
                "N_DATA_WORKERS": "0",
                "PREDICTIONS_OUTPUT_DIR": output_dir,
                "MISSING_LIMIT": "0",
            },
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


def test_predict_validate_combined_dataloader_results_from_configuration(lib_path: str) -> None:
    """Tests combined validation results with multiple prediction validation datasets."""
    configuration_file = "configs/language/tests/offline/pubmedqa_combine.yaml"

    with tempfile.TemporaryDirectory() as temp_dir:
        predictions_dir = os.path.join(temp_dir, "predictions")
        logs_dir = os.path.join(temp_dir, "logs")

        with mock.patch.dict(
            os.environ,
            {
                "N_RUNS": "1",
                "BATCH_SIZE": "2",
                "N_DATA_WORKERS": "0",
                "PREDICTIONS_OUTPUT_DIR": predictions_dir,
                "OUTPUT_ROOT": logs_dir,
                "MISSING_LIMIT": "0",
            },
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

        with open(_find_results_file(logs_dir), "r") as file:
            results = json.load(file)

        assert len(results["metrics"]["val"]) == 1
        metric_statistics = next(iter(results["metrics"]["val"][0].values()))
        assert len(metric_statistics["values"]) == 2
        assert metric_statistics["mean"] == pytest.approx(
            statistics.mean(metric_statistics["values"])
        )


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mocks external dependencies to avoid API calls and downloads."""

    def _fake_completion(
        model: str = "dummy-model", messages: List | None = None, **kwargs
    ) -> List[Dict[str, Any]]:
        return [
            {
                "choices": [
                    {"message": {"content": 'some text {"answer": "yes"} end', "role": "assistant"}}
                ]
            }
        ] * len(messages or [])

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
            lambda **_kwargs: _fake_completion(**_kwargs),
        ),
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}),
        mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy-key"}),
    ):
        yield


@pytest.fixture(autouse=True)
def skip_dataset_validation() -> None:
    """Mocks the validation step of the datasets."""
    datasets.PubMedQA.validate = mock.MagicMock(return_value=None)


def _find_results_file(output_dir: str) -> str:
    """Returns the path of the generated results file."""
    result_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file == "results.json":
                result_files.append(os.path.join(root, file))

    assert len(result_files) == 1
    return result_files[0]
