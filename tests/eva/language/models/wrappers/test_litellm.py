"""LiteLLM wrapper tests."""

import pytest

from eva.language.models import LiteLLMTextModel

DUMMY_RESPONSE = {"choices": [{"message": {"content": "Test response"}}]}


def test_generate(model_instance):
    """Test that the generate method returns the expected dummy response."""
    prompt = "Hello, world!"
    result = model_instance.generate(prompt)
    assert result == "Test response"


@pytest.fixture
def fake_completion(monkeypatch):
    """Fixture to override `completion` function and set a dummy OPENAI_API_KEY."""

    def _fake_completion(model, messages, **kwargs):
        assert isinstance(messages, list)
        assert messages and messages[0]["role"] == "user"
        return DUMMY_RESPONSE

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    monkeypatch.setattr("eva.language.models.wrappers.litellm.completion", _fake_completion)
    return _fake_completion


@pytest.fixture
def model_instance(fake_completion):
    """Fixture to instantiate the LiteLLMTextModel with a valid model name.

    Using a valid model name (like 'openai/gpt-3.5-turbo') helps pass the provider lookup.
    """
    return LiteLLMTextModel("openai/gpt-3.5-turbo", model_kwargs={"temperature": 0.7})
