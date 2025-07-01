"""HuggingFaceModel wrapper tests."""

from unittest.mock import MagicMock, patch

import pytest

from eva.language.models import HuggingFaceTextModel


@pytest.mark.parametrize(
    "model_name_or_path, prompt, generate_kwargs, expect_deterministic",
    [
        (
            "sshleifer/tiny-gpt2",
            "Once upon a time",
            {"max_length": 30, "do_sample": False},
            True,
        ),
        (
            "sshleifer/tiny-gpt2",
            "In a galaxy far, far away",
            {"max_length": 30, "do_sample": True},
            False,
        ),
    ],
)
def test_real_small_hf_model_generation(
    model_name_or_path: str,
    prompt: str,
    generate_kwargs: dict,
    expect_deterministic: bool,
):
    """Test HuggingFace model generation with mocked pipeline.

    Tests the wrapper correctly handles generation parameters and returns
    expected output format for deterministic vs non-deterministic generation.
    """
    mock_pipeline = MagicMock()

    if expect_deterministic:
        mock_pipeline.return_value = [[{"generated_text": f"{prompt} generated"}]]
    else:
        mock_pipeline.side_effect = [
            [[{"generated_text": f"{prompt} generated 1"}]],
            [[{"generated_text": f"{prompt} generated 2"}]],
        ]

    with patch("eva.language.models.wrappers.huggingface.pipeline", return_value=mock_pipeline):
        model = HuggingFaceTextModel(
            model_name_or_path=model_name_or_path,
            task="text-generation",
            generation_kwargs=generate_kwargs,
        )

        output1 = model([prompt])[0]
        output2 = model([prompt])[0]

        assert isinstance(output1, str) and output1, "First output should be a non-empty string."
        assert isinstance(output2, str) and output2, "Second output should be a non-empty string."

        if expect_deterministic:
            assert output1 == output2, "Outputs should be identical when do_sample is False."
        else:
            # For non-deterministic, outputs should differ
            assert output1 != output2, "Outputs should differ when do_sample is True."
