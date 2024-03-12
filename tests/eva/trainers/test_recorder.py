"""Tests the SessionRecorder class."""

import tempfile
from typing import Any, Dict, List

import pytest
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT

from eva.trainers import _recorder

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


@pytest.fixture(scope="function")
def session_recorder() -> _recorder.SessionRecorder:
    """`SessionRecorder` fixture."""
    with tempfile.TemporaryDirectory() as tempdir:
        return _recorder.SessionRecorder(output_dir=tempdir)
