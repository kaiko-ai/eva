"""LiteLLM wrapper tests."""

import pytest

from eva.language.models import LiteLLMTextModel

DUMMY_RESPONSE = {"choices": [{"message": {"content": "Test response"}}]}


def test_generate(model_instance):
    """Test that the generate method returns the expected dummy response."""
    prompts = ["Hello, world!"]
    result = model_instance(prompts)
    assert result == ["Test response"]


@pytest.fixture
def fake_completion(monkeypatch):
    """Fixture to override `batch_completion` function and set a dummy OPENAI_API_KEY."""

    def _fake_batch_completion(**_kwargs):
        return [DUMMY_RESPONSE]

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    monkeypatch.setattr(
        "eva.language.models.wrappers.litellm.batch_completion", _fake_batch_completion
    )
    return _fake_batch_completion


@pytest.fixture
def model_instance(fake_completion):  # noqa: ARG001
    """Fixture to instantiate the LiteLLMTextModel with a valid model name.

    Using a valid model name (like 'openai/gpt-3.5-turbo') helps pass provider lookup.
    fake_completion dependency ensures mocking is set up before model creation.
    """
    return LiteLLMTextModel("openai/gpt-3.5-turbo", model_kwargs={"temperature": 0.7})
