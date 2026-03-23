"""HuggingFace multimodal wrapper tests."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision import tv_tensors

from eva.language.data.messages import UserMessage
from eva.multimodal.models.typings import TextImageBatch
from eva.multimodal.models.wrappers.huggingface import HuggingFaceModel


@pytest.mark.parametrize(
    "model_name, model_class, with_image",
    [
        ("llava-hf/llava-1.5-7b-hf", "LlavaForConditionalGeneration", True),
        ("llava-hf/llava-1.5-7b-hf", "LlavaForConditionalGeneration", False),
    ],
)
def test_huggingface_model_generation(model_name: str, model_class: str, with_image: bool):
    """Test HuggingFace multimodal model generation with mocked components."""
    mock_processor = MagicMock()
    mock_processor.chat_template = "template"
    mock_processor.apply_chat_template.return_value = "formatted text"
    mock_processor.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_processor.batch_decode.side_effect = [[""], ["Generated response"]]

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    setattr(mock_transformers, model_class, MagicMock())
    getattr(mock_transformers, model_class).from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path=model_name,
            model_class=model_class,
            generation_kwargs={"max_new_tokens": 50},
        )
        model.configure_model()

        # Always create an image tensor, even if not used
        image = tv_tensors.Image(torch.rand(3, 224, 224))
        batch = TextImageBatch(
            text=[[UserMessage(content="Describe this")]],
            images=[[image]],
            target=None,
            metadata={},
        )

        result = model(batch)
        assert isinstance(result, dict)
        assert "generated_text" in result
        assert "input_text" in result
        assert "output_ids" in result
        assert "attention_mask" in result
        assert result["generated_text"] == ["Generated response"]
        assert result["input_text"] == [""]
        assert mock_model.generate.called


def test_format_inputs_with_image():
    """Test format_inputs correctly handles image inputs."""
    mock_processor = MagicMock()
    mock_processor.chat_template = "template"
    mock_processor.apply_chat_template.return_value = "formatted text"
    mock_processor.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.rand(1, 3, 224, 224),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.LlavaForConditionalGeneration = MagicMock()
    mock_transformers.LlavaForConditionalGeneration.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path="test-model",
            model_class="LlavaForConditionalGeneration",
        )
        model.configure_model()

        image = tv_tensors.Image(torch.rand(3, 224, 224))
        batch = TextImageBatch(
            text=[[UserMessage(content="Test")]],
            images=[[image]],
            target=None,
            metadata={},
        )

        formatted = model.format_inputs(batch)

        mock_processor.assert_called_with(
            text=["formatted text"],
            image=[[image]],
            return_tensors="pt",
        )
        assert "input_ids" in formatted


def test_decode_ids():
    """Test _decode_ids correctly decodes both input and output."""
    mock_processor = MagicMock()
    mock_processor.batch_decode.side_effect = [["Input text"], ["Output text"]]

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.LlavaForConditionalGeneration = MagicMock()
    mock_transformers.LlavaForConditionalGeneration.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path="test-model",
            model_class="LlavaForConditionalGeneration",
        )
        model.configure_model()

        output = torch.tensor([[1, 2, 3, 4, 5, 6]])
        instruction_length = 3

        decoded_input, decoded_output = model.model._decode_ids(output, instruction_length)

        assert decoded_input == ["Input text"]
        assert decoded_output == ["Output text"]
        assert mock_processor.batch_decode.call_count == 2


def test_chat_template_passed_to_language_wrapper():
    """Test that chat_template is passed to the underlying language wrapper."""
    custom_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    mock_processor = MagicMock()
    mock_processor.chat_template = None

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.LlavaForConditionalGeneration = MagicMock()
    mock_transformers.LlavaForConditionalGeneration.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path="test-model",
            model_class="LlavaForConditionalGeneration",
            chat_template=custom_template,
        )
        model.configure_model()

        assert model.chat_template == custom_template
        assert model.model.chat_template == custom_template
        assert mock_processor.chat_template == custom_template


def test_chat_template_none_uses_default():
    """Test that when chat_template is None, processor's default template is used."""
    default_template = "default_template"

    mock_processor = MagicMock()
    mock_processor.chat_template = default_template

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
    mock_transformers.LlavaForConditionalGeneration = MagicMock()
    mock_transformers.LlavaForConditionalGeneration.from_pretrained.return_value = mock_model

    with (
        patch.dict("sys.modules", {"transformers": mock_transformers}),
        patch("eva.language.models.wrappers.huggingface.transformers", mock_transformers),
    ):
        model = HuggingFaceModel(
            model_name_or_path="test-model",
            model_class="LlavaForConditionalGeneration",
            chat_template=None,
        )
        model.configure_model()

        assert model.chat_template is None
        assert mock_processor.chat_template == default_template
