"""Unit tests for G-Eval LLM Judge implementation."""

from eva.language.metrics.llm_judge.g_eval.judge import GEvalJudge
from eva.language.models import wrappers
from eva.language.models.typings import PredictionBatch


def test_g_eval_judge_evaluate_flow(dummy_judge_model: wrappers.LanguageModel) -> None:
    """Test the main evaluation flow with a dummy model."""
    judge = GEvalJudge(
        model=dummy_judge_model,
        evaluation_steps=["Check factual accuracy", "Evaluate completeness"],
        score_range=(0, 10),
        score_explanation="where higher is better",
    )

    batch = PredictionBatch(
        prediction=["The capital of France is Paris.", "The capital of Spain is Madrid."],
        target=["Paris", "Madrid"],
        text=None,
        metadata=None,
    )

    scores = judge.evaluate(batch)

    assert len(scores) == 2
    assert scores[0] == 8
    assert scores[1] == 5
