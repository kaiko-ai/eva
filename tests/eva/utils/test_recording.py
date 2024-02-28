"""Tests recorder functions."""

import json
import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import patch

from eva.utils import recording


def test_record_results() -> None:
    """Tests if results dict is written to disk."""
    _expected_keys = ["start_time", "end_time", "duration", "metrics"]

    start_time = datetime.now()
    end_time = datetime.now()
    results = {}

    with tempfile.TemporaryDirectory() as output_dir:
        results_path = os.path.join(output_dir, "results.json")
        recording.record_results(results, results_path, start_time, end_time)
        with open(f"{output_dir}/results.json", "r") as f:
            data = json.load(f)
    assert sorted(data.keys()) == sorted(_expected_keys)


def test_get_evaluation_id() -> None:
    """Tests if evaluation ID is generated correctly."""
    _test_args = [
        ["-config", "configs/vision/tests/offline/patch_camelyon.yaml"],
        ["-config", "configs/vision/tests/offline/patches.yaml"],
        ["-config", ""],
    ]

    evaluation_ids = []
    for test_args in _test_args:
        with patch.object(sys, "argv", test_args):
            evaluation_ids.append(recording.get_evaluation_id())

    assert len(evaluation_ids[0]) == 30
    assert len(evaluation_ids[1]) == 30
    assert len(evaluation_ids[2]) == 22
    assert evaluation_ids[0] != evaluation_ids[1]
