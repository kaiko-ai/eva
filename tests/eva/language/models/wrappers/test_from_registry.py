"""Tests for language from registry model wrapper."""

import os
from unittest import mock

import pytest

from eva.language.models import wrappers


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("anthropic/claude-3-7-sonnet-20250219", wrappers.LiteLLMModel),
    ],
)
def test_load_model(model_name: str, model_class: type):
    """Test loading a model from the registry."""
    with mock.patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "test_key",
        },
    ):
        model = wrappers.ModelFromRegistry(model_name)

    assert isinstance(model, wrappers.ModelFromRegistry)
    assert isinstance(model.model, model_class)
