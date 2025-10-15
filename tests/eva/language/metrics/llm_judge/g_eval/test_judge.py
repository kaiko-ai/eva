"""Tests for the G-Eval LLM judge."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from eva.language.data.messages import UserMessage
from eva.language.metrics.llm_judge.g_eval.judge import GEvalJudge
from eva.language.models.typings import PredictionBatch, TextBatch


def _build_batch(predictions: list[str], targets: list[str]) -> PredictionBatch[list[str]]:
    """Helper to build a prediction batch."""
    return PredictionBatch(prediction=predictions, target=targets, text=None, metadata=None)


def test_evaluate_uses_rendered_prompts_and_returns_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure evaluate renders prompts, calls the model, and returns parsed scores."""
    model = MagicMock()
    model.return_value = {"generated_text": ["{}", "{}"]}
    judge = GEvalJudge(
        model=model,
        evaluation_steps=["Check structure", "Compare facts"],
        score_range=(0, 5),
        score_explanation="higher means better",
    )
    parsed_outputs = [(3, "solid reasoning"), (1, "missing detail")]
    monkeypatch.setattr(judge, "_parse_output", MagicMock(side_effect=parsed_outputs))

    batch = _build_batch(["Answer one", "Answer two"], ["Truth one", "Truth two"])
    scores = judge.evaluate(batch)

    assert scores == [3, 1]
    model.assert_called_once()
    called_batch: TextBatch[Any] = model.call_args.args[0]
    assert called_batch.target is None
    assert called_batch.metadata is None
    assert len(called_batch.text) == 2
    first_prompt = called_batch.text[0][0]
    assert isinstance(first_prompt, UserMessage)
    content = first_prompt.content
    assert "1. Check structure" in content
    assert "Model Response:\n        Answer one" in content
    assert "Ground Truth:\n        Truth one" in content


@pytest.mark.parametrize(
    ("extracted_json", "expected"),
    [
        ({"score": 4, "reason": "clear comparison"}, (4, "clear comparison")),
        ({"score": "0", "reason": "no alignment"}, (0, "no alignment")),
        (None, (None, "Failed to extract JSON from model output.")),
    ],
)
def test_parse_output_handles_json(
    monkeypatch: pytest.MonkeyPatch,
    extracted_json: dict[str, Any] | None,
    expected: tuple[int | None, str],
) -> None:
    """Verify _parse_output converts extracted JSON into scores."""
    judge = GEvalJudge(model=MagicMock(), evaluation_steps=["Review"])
    monkeypatch.setattr(
        "eva.language.metrics.llm_judge.g_eval.judge.json_utils.extract_json",
        lambda output: extracted_json,
    )

    assert judge._parse_output("raw") == expected


@pytest.mark.parametrize(
    ("extracted_json", "error_message"),
    [
        ({"reason": "no score"}, "missing required"),
        ({"score": 11, "reason": "too high"}, "Score 11 is out of the expected range"),
    ],
)
def test_parse_output_raises_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    extracted_json: dict[str, Any],
    error_message: str,
) -> None:
    """Ensure _parse_output raises when JSON is invalid or out of range."""
    judge = GEvalJudge(model=MagicMock(), evaluation_steps=["Review"], score_range=(0, 10))
    monkeypatch.setattr(
        "eva.language.metrics.llm_judge.g_eval.judge.json_utils.extract_json",
        lambda output: extracted_json,
    )

    with pytest.raises(ValueError, match=error_message):
        judge._parse_output("raw")


@pytest.mark.parametrize(
    ("input_model", "expected_name"),
    [
        (None, "google/gemini-2.5-flash-lite"),
        ("custom/model", "custom/model"),
    ],
)
def test_load_model_uses_registry(
    monkeypatch: pytest.MonkeyPatch,
    input_model: str | None,
    expected_name: str,
) -> None:
    """Verify _load_model loads from the registry when given None or a string."""
    judge = GEvalJudge(model=MagicMock(), evaluation_steps=["Review"])
    captured = {}

    def fake_registry(name: str) -> str:
        captured["name"] = name
        return f"loaded:{name}"

    monkeypatch.setattr(
        "eva.language.metrics.llm_judge.g_eval.judge.wrappers.ModelFromRegistry",
        fake_registry,
    )

    loaded = judge._load_model(input_model)
    assert loaded == f"loaded:{expected_name}"
    assert captured["name"] == expected_name


def test_load_model_returns_existing_instance() -> None:
    """Ensure an existing model instance is returned unchanged."""
    existing_model = MagicMock()
    judge = GEvalJudge(model=existing_model, evaluation_steps=["Review"])

    assert judge._load_model(existing_model) is existing_model
