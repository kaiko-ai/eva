"""Tests regarding eva's CLI commands on multimodal datasets."""

import os
import random
import tempfile
from typing import Any, Dict, List
from unittest import mock
from unittest.mock import patch

import pytest

from eva.language.models.typings import PredictionBatch
from eva.multimodal.data import datasets
from tests.eva import _cli

BATCH_SIZE = 2


@pytest.mark.parametrize(
    "configuration_file",
    [
        "configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml",
        "configs/multimodal/pathology/offline/multiple_choice/patch_camelyon.yaml",
        "configs/multimodal/pathology/online/free_form/quilt_vqa.yaml",
        "configs/multimodal/pathology/offline/free_form/quilt_vqa.yaml",
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
    "configuration_file, command, data_root",
    [
        # (
        #     "configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml",
        #     "validate",
        #     "vision/datasets/patch_camelyon",
        # ),
        # (
        #     "configs/multimodal/pathology/online/multiple_choice/patch_camelyon.yaml",
        #     "test",
        #     "vision/datasets/patch_camelyon",
        # ),
        (
            "configs/multimodal/pathology/online/free_form/quilt_vqa.yaml",
            "test",
            "multimodal/datasets/quilt_vqa",
        ),
    ],
)
def test_validate_from_configuration(
    configuration_file: str, command: str, lib_path: str, assets_path: str, data_root: str
) -> None:
    """Tests CLI `validate` & `test` commands with a given configuration file."""
    with mock.patch.dict(
        os.environ,
        {
            "N_RUNS": "1",
            "BATCH_SIZE": f"{BATCH_SIZE}",
            "DATA_ROOT": os.path.join(assets_path, data_root),
            "NUM_SAMPLES": "null",
        },
    ):
        _cli.run_cli_from_main(
            cli_args=[
                command,
                "--config",
                os.path.join(lib_path, configuration_file),
            ]
        )


@pytest.mark.parametrize(
    "configuration_file, command, data_root",
    [
        (
            "configs/multimodal/pathology/offline/multiple_choice/patch_camelyon.yaml",
            "validate",
            "vision/datasets/patch_camelyon",
        ),
        (
            "configs/multimodal/pathology/offline/free_form/quilt_vqa.yaml",
            "test",
            "multimodal/datasets/quilt_vqa",
        ),
    ],
)
def test_predict_validate_from_configuration(
    configuration_file: str, command: str, lib_path: str, assets_path: str, data_root: str
) -> None:
    """Tests CLI `predict` and `validate` / `test` commands with a given configuration file."""
    with tempfile.TemporaryDirectory() as output_dir:
        with mock.patch.dict(
            os.environ,
            {
                "N_RUNS": "1",
                "PREDICT_BATCH_SIZE": "2",
                "PREDICTIONS_OUTPUT_DIR": output_dir,
                "DATA_ROOT": os.path.join(assets_path, data_root),
                "NUM_SAMPLES": "null",
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
                    command,
                    "--config",
                    os.path.join(lib_path, configuration_file),
                ]
            )


@pytest.fixture(autouse=True)
def skip_dataset_validation() -> None:
    """Mocks the validation step of the datasets."""
    datasets.PatchCamelyon.validate = mock.MagicMock(return_value=None)
    datasets.QuiltVQA.validate = mock.MagicMock(return_value=None)


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

    def _fake_judge_evaluate(batch: PredictionBatch) -> List[int | None]:
        scores: List[int | None] = [random.randint(1, 5) for _ in batch.prediction]
        scores[-1] = None  # simulate a missing score
        return scores

    with (
        patch(
            "eva.language.models.wrappers.litellm.batch_completion",
            lambda **_kwargs: _fake_completion(**_kwargs),
        ),
        patch(
            "eva.language.metrics.llm_judge.g_eval.judge.GEvalJudge.evaluate",
            side_effect=_fake_judge_evaluate,
        ),
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}),
        mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy-key"}),
        mock.patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"}),
    ):
        yield
