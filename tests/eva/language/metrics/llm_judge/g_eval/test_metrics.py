"""Unit tests for G-Eval metric implementations."""

from unittest.mock import Mock, patch

import pytest
import torch

from eva.language.metrics.llm_judge.g_eval.metrics import GEvalCorrectness
from eva.language.models import wrappers


@pytest.fixture
def g_eval_metric(dummy_judge_model: wrappers.LanguageModel):
    """Create a G-Eval correctness metric with a mocked judge."""
    with patch("eva.language.metrics.llm_judge.g_eval.metrics.GEvalJudge") as mock_judge_class:
        mock_judge = Mock()
        # Mock evaluate to return scores based on batch size
        mock_judge.evaluate.side_effect = lambda batch: [5] * len(batch.prediction)
        mock_judge_class.return_value = mock_judge

        metric = GEvalCorrectness(model=dummy_judge_model)
        metric.mock_judge = mock_judge  # Store for test access
        yield metric


def test_g_eval_correctness_initialization(dummy_judge_model: wrappers.LanguageModel) -> None:
    """Test that GEvalCorrectness initializes correctly."""
    with patch("eva.language.metrics.llm_judge.g_eval.metrics.GEvalJudge"):
        metric = GEvalCorrectness(model=dummy_judge_model)

        assert metric.judge is not None
        assert metric.total == 0.0
        assert metric.count == 0


def test_g_eval_correctness_update(g_eval_metric: GEvalCorrectness) -> None:
    """Test updating the metric with predictions and targets."""
    preds = ["Paris is the capital.", "Madrid is the capital.", "Berlin is the capital."]
    targets = ["Paris", "Madrid", "Berlin"]

    g_eval_metric.update(preds, targets)

    assert g_eval_metric.count == 3
    assert g_eval_metric.total == 15.0  # 5 + 5 + 5


def test_g_eval_correctness_compute(g_eval_metric: GEvalCorrectness) -> None:
    """Test computing the final metric value."""
    preds = ["Paris is the capital.", "Madrid is the capital.", "Berlin is the capital."]
    targets = ["Paris", "Madrid", "Berlin"]

    g_eval_metric.update(preds, targets)
    result = g_eval_metric.compute()

    assert isinstance(result, torch.Tensor)
    assert result.item() == pytest.approx(5.0)  # (5 + 5 + 5) / 3


def test_g_eval_correctness_multiple_updates(g_eval_metric: GEvalCorrectness) -> None:
    """Test multiple updates accumulate correctly."""
    # First update
    g_eval_metric.update(["Answer 1"], ["Target 1"])
    assert g_eval_metric.count == 1
    assert g_eval_metric.total == 5.0

    # Second update
    g_eval_metric.update(["Answer 2", "Answer 3"], ["Target 2", "Target 3"])
    assert g_eval_metric.count == 3
    assert g_eval_metric.total == 15.0  # 5 + 5 + 5

    result = g_eval_metric.compute()
    assert result.item() == pytest.approx(5.0)


def test_g_eval_correctness_score_range() -> None:
    """Test that the metric is configured with the correct score range."""
    assert GEvalCorrectness._score_range == (1, 5)


def test_g_eval_correctness_evaluation_steps() -> None:
    """Test that evaluation steps are properly defined."""
    assert len(GEvalCorrectness._evaluation_steps) == 4
    assert (
        "Read the Model Response and Ground Truth carefully" in GEvalCorrectness._evaluation_steps
    )


def test_g_eval_correctness_scoring_criteria() -> None:
    """Test that scoring criteria is properly defined."""
    criteria = GEvalCorrectness._scoring_criteria
    assert "5 (Excellent)" in criteria
    assert "1 (Very Poor)" in criteria
