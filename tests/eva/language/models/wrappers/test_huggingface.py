"""HuggingFaceModel wrapper tests."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from eva.language.data.messages import UserMessage
from eva.language.models import HuggingFaceModel
from eva.language.models.typings import TextBatch


@pytest.fixture
def mock_processor():
    """Create a mock processor."""
    processor = MagicMock()
    processor.chat_template = "mock_template"
    processor.apply_chat_template = MagicMock(return_value="formatted prompt")
    processor.batch_decode = MagicMock(
        side_effect=lambda x, **kwargs: ["decoded text"] * x.shape[0]
    )

    # Mock the processor call to return a BatchEncoding-like object
    mock_encoding = MagicMock()
    mock_encoding.__getitem__ = lambda self, key: torch.tensor([[1, 2, 3]])
    mock_encoding.get = MagicMock(return_value=torch.tensor([[1, 1, 1]]))
    mock_encoding.to = MagicMock(return_value=mock_encoding)
    processor.return_value = mock_encoding

    return processor


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5, 6]]))
    return model


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
def test_huggingface_model_generation(
    model_name_or_path: str,
    prompt: str,
    generate_kwargs: dict,
    expect_deterministic: bool,
    mock_processor,
    mock_model,
):
    """Test HuggingFace model generation with mocked model and tokenizer.

    Tests the wrapper correctly handles generation parameters and returns
    expected output format for deterministic vs non-deterministic generation.
    """
    if expect_deterministic:
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
    else:
        mock_model.generate.side_effect = [
            torch.tensor([[1, 2, 3, 4, 5, 6]]),
            torch.tensor([[1, 2, 3, 7, 8, 9]]),
        ]

    # Different decoded outputs for non-deterministic case
    if not expect_deterministic:
        mock_processor.batch_decode.side_effect = [
            ["input text"],
            [f"{prompt} generated 1"],
            ["input text"],
            [f"{prompt} generated 2"],
        ]
    else:
        mock_processor.batch_decode.side_effect = [
            ["input text"],
            [f"{prompt} generated"],
            ["input text"],
            [f"{prompt} generated"],
        ]

    with (
        patch(
            "eva.language.models.wrappers.huggingface.transformers.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ),
        patch(
            "eva.language.models.wrappers.huggingface.transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
        patch("eva.language.models.wrappers.huggingface.transformers") as mock_transformers,
    ):
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModelForCausalLM = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        model = HuggingFaceModel(
            model_name_or_path=model_name_or_path,
            model_class="AutoModelForCausalLM",
            generation_kwargs=generate_kwargs,
        )
        model.configure_model()

        batch = TextBatch(text=[[UserMessage(content=prompt)]], target=None, metadata={})
        output1 = model(batch)
        output2 = model(batch)

        assert isinstance(output1, dict) and "generated_text" in output1
        assert isinstance(output2, dict) and "generated_text" in output2

        if expect_deterministic:
            assert (
                output1["generated_text"] == output2["generated_text"]
            ), "Outputs should be identical when do_sample is False."
        else:
            assert (
                output1["generated_text"] != output2["generated_text"]
            ), "Outputs should differ when do_sample is True."


def test_huggingface_model_invalid_class():
    """Test that an invalid model class raises ValueError."""
    with patch(
        "eva.language.models.wrappers.huggingface.transformers.AutoProcessor.from_pretrained",
        return_value=MagicMock(),
    ):
        with pytest.raises(ValueError, match="not found in transformers"):
            model = HuggingFaceModel(
                model_name_or_path="some-model",
                model_class="NonExistentModelClass",
            )
            model.configure_model()


def test_huggingface_model_no_generate_support(mock_processor):
    """Test that a model without generate method raises ValueError."""
    mock_model = MagicMock(spec=[])  # Model without generate attribute

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        with pytest.raises(ValueError, match="does not support generation"):
            model = HuggingFaceModel(
                model_name_or_path="some-model",
                model_class="AutoModelForCausalLM",
            )
            model.configure_model()


def test_chat_template_applied_to_processor():
    """Test that custom chat_template is applied to the processor."""
    custom_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    mock_processor = MagicMock()
    mock_processor.chat_template = None  # Initially no template

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path="some-model",
            model_class="AutoModelForCausalLM",
            chat_template=custom_template,
        )
        model.configure_model()

        assert model.chat_template == custom_template
        assert mock_processor.chat_template == custom_template


def test_chat_template_none_uses_processor_default():
    """Test that when chat_template is None, processor's default template is used."""
    default_template = "default_template"

    mock_processor = MagicMock()
    mock_processor.chat_template = default_template

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.AutoModelForCausalLM = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path="some-model",
            model_class="AutoModelForCausalLM",
            chat_template=None,
        )
        model.configure_model()

        assert model.chat_template is None
        assert mock_processor.chat_template == default_template
