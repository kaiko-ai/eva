"""HuggingFaceModel wrapper tests."""

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
    """Integration test using a real, small Hugging Face model for text generation.

    The test instantiates the HuggingFaceTextModel, generates text for a given prompt,
    and asserts that the output is a non-empty string. If sampling is disabled,
    repeated calls should return identical outputs.
    """
    model = HuggingFaceTextModel(model_name_or_path=model_name_or_path, task="text-generation")

    output1 = model.generate(prompt, **generate_kwargs)
    output2 = model.generate(prompt, **generate_kwargs)

    assert isinstance(output1, str) and output1, "First output should be a non-empty string."
    assert isinstance(output2, str) and output2, "Second output should be a non-empty string."

    if expect_deterministic:
        assert output1 == output2, "Outputs should be identical when do_sample is False."
