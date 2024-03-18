"""Tests the SessionRecorder class."""

import os
from typing import Any, Dict, List
from unittest import mock

import pytest
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT

from eva.core.trainers import _recorder

_RELATIVE_TOLERANCE = 1e-3
"""The test relative tolerance."""

SESSION_RESULTS_ONE = [
    [{"val/AverageLoss": 1.0, "val/MulticlassAccuracy": 0.1}],
    [{"val/AverageLoss": 2.0, "val/MulticlassAccuracy": 0.2}],
]
EXPECTED_ONE = {
    "val": [
        {
            "val/AverageLoss": {
                "mean": 1.5,
                "stdev": pytest.approx(0.7071, rel=_RELATIVE_TOLERANCE),
                "values": [1.0, 2.0],
            },
            "val/MulticlassAccuracy": {
                "mean": pytest.approx(0.1500, rel=_RELATIVE_TOLERANCE),
                "stdev": pytest.approx(0.0707, rel=_RELATIVE_TOLERANCE),
                "values": [0.1, 0.2],
            },
        }
    ],
    "test": [],
}
"""Test features one."""

SESSION_RESULTS_TWO = [
    [
        {"val/AverageLoss": 1.0, "val/MulticlassAccuracy": 0.1},
        {"val/AverageLoss": 3.0, "val/MulticlassAccuracy": 0.3},
    ],
    [
        {"val/AverageLoss": 2.0, "val/MulticlassAccuracy": 0.2},
        {"val/AverageLoss": 4.0, "val/MulticlassAccuracy": 0.4},
    ],
]
EXPECTED_TWO = {
    "val": [
        {
            "val/AverageLoss": {
                "mean": 1.5,
                "stdev": pytest.approx(0.7071067811865476, rel=_RELATIVE_TOLERANCE),
                "values": [1.0, 2.0],
            },
            "val/MulticlassAccuracy": {
                "mean": pytest.approx(0.15000000000000002, rel=_RELATIVE_TOLERANCE),
                "stdev": pytest.approx(0.07071067811865477, rel=_RELATIVE_TOLERANCE),
                "values": [0.1, 0.2],
            },
        },
        {
            "val/AverageLoss": {
                "mean": 3.5,
                "stdev": pytest.approx(0.7071067811865476, rel=_RELATIVE_TOLERANCE),
                "values": [3.0, 4.0],
            },
            "val/MulticlassAccuracy": {
                "mean": pytest.approx(0.35, rel=_RELATIVE_TOLERANCE),
                "stdev": pytest.approx(0.07071067811865478, rel=_RELATIVE_TOLERANCE),
                "values": [0.3, 0.4],
            },
        },
    ],
    "test": [],
}
"""Test features two."""


@pytest.mark.parametrize(
    "session_results, expected",
    [
        # (SESSION_RESULTS_ONE, EXPECTED_ONE),
        (SESSION_RESULTS_TWO, EXPECTED_TWO),
    ],
)
def test_session_recorder(
    session_recorder: _recorder.SessionRecorder,
    session_results: List[_EVALUATE_OUTPUT],
    expected: Dict[str, Any],
) -> None:
    """Tests the average_loss_metric metric."""

    def _calculate_metric() -> None:
        for run_results in session_results:
            session_recorder.update(validation_scores=run_results)  # type: ignore
        actual = session_recorder.compute()
        assert actual == expected

    _calculate_metric()
    session_recorder.reset()
    _calculate_metric()


def test_save_config(session_recorder: _recorder.SessionRecorder, tmp_path_factory):
    """Tests if the input .yaml configuration file is saved."""
    # Save a fake .yaml configuration file
    config_dir = tmp_path_factory.mktemp("config")
    input_config_path = os.path.join(config_dir, "config.yaml")
    with open(input_config_path, "w") as file:
        file.write("test: true")

    # Invoke the save method and check if the file is saved to the output directory
    with mock.patch.object(
        _recorder.SessionRecorder, "config_path", new_callable=mock.PropertyMock
    ) as mock_config_path:
        mock_config_path.return_value = input_config_path
        assert isinstance(session_recorder.config_path, str)

        output_config_path = os.path.join(
            session_recorder._output_dir, os.path.basename(session_recorder.config_path)
        )

        assert not os.path.isfile(output_config_path)
        session_recorder.save()
        assert os.path.isfile(output_config_path)


@pytest.fixture(scope="function")
def session_recorder(tmp_path_factory) -> _recorder.SessionRecorder:
    """`SessionRecorder` fixture."""
    output_dir = tmp_path_factory.mktemp("output")
    return _recorder.SessionRecorder(output_dir=output_dir)
