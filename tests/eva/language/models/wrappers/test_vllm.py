"""vLLM language wrapper tests."""

from unittest.mock import MagicMock, patch

import pytest

from eva.language.data.messages import UserMessage
from eva.language.models.typings import TextBatch
from eva.language.utils.imports import is_vllm_available

if not is_vllm_available():
    pytest.skip("vLLM not available", allow_module_level=True)

from eva.language.models.wrappers.vllm import VllmModel


@pytest.fixture
def mock_llm():
    """Mock vLLM LLM instance."""
    mock_llm_instance = MagicMock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="Generated response")]
    mock_llm_instance.generate.return_value = [mock_output]
    return mock_llm_instance


@pytest.fixture
def mock_tokenizer():
    """Mock AutoTokenizer with chat template."""
    mock_tok = MagicMock()
    mock_tok.chat_template = "template"
    mock_tok.apply_chat_template.return_value = "formatted prompt"
    return mock_tok


@pytest.fixture
def model_instance(mock_llm, mock_tokenizer):
    """Fixture to instantiate the VllmModel with mocked components."""
    with (
        patch.object(VllmModel, "load_model", return_value=mock_llm),
        patch.object(VllmModel, "load_tokenizer", return_value=mock_tokenizer),
    ):
        model = VllmModel(
            model_name_or_path="test-model",
            system_prompt="You are a helpful assistant.",
            model_kwargs={"tensor_parallel_size": 1},
            generation_kwargs={"temperature": 0.0},
        )
        model.configure_model()
        yield model


def test_initialization():
    """Test VllmModel initialization with default and custom kwargs."""
    model = VllmModel(
        model_name_or_path="test-model",
        model_kwargs={"tensor_parallel_size": 2},
        generation_kwargs={"temperature": 0.5},
    )

    assert model.model_name_or_path == "test-model"
    assert model.model_kwargs["tensor_parallel_size"] == 2
    assert model.model_kwargs["max_model_len"] == 32768  # default preserved
    assert model.generation_kwargs["temperature"] == 0.5


def test_default_kwargs_preserved():
    """Test that default kwargs are preserved when not overridden."""
    model = VllmModel(model_name_or_path="test-model")

    assert model.model_kwargs["max_model_len"] == 32768
    assert model.model_kwargs["gpu_memory_utilization"] == 0.95
    assert model.model_kwargs["tensor_parallel_size"] == 1
    assert model.model_kwargs["dtype"] == "auto"
    assert model.model_kwargs["trust_remote_code"] is True
    assert model.generation_kwargs["temperature"] == 0.0
    assert model.generation_kwargs["top_p"] == 1.0
    assert model.generation_kwargs["top_k"] == -1
    assert model.generation_kwargs["n"] == 1


def test_configure_model_loads_model_and_tokenizer(mock_llm, mock_tokenizer):
    """Test that configure_model loads model and tokenizer."""
    with (
        patch.object(VllmModel, "load_model", return_value=mock_llm) as load_model_mock,
        patch.object(VllmModel, "load_tokenizer", return_value=mock_tokenizer) as load_tok_mock,
    ):
        model = VllmModel(model_name_or_path="test-model")

        model.configure_model()

        assert model.model is mock_llm
        assert model.tokenizer is mock_tokenizer
        load_model_mock.assert_called_once()
        load_tok_mock.assert_called_once()


def test_configure_model_idempotent(mock_llm, mock_tokenizer):
    """Test that configure_model only loads once."""
    with (
        patch.object(VllmModel, "load_model", return_value=mock_llm) as load_model_mock,
        patch.object(VllmModel, "load_tokenizer", return_value=mock_tokenizer) as load_tok_mock,
    ):
        model = VllmModel(model_name_or_path="test-model")
        model.configure_model()
        model.configure_model()  # second call

        assert load_model_mock.call_count == 1
        assert load_tok_mock.call_count == 1


def test_tokenizer_without_chat_template_raises():
    """Test that loading a tokenizer without chat template raises NotImplementedError."""
    mock_tokenizer_no_template = MagicMock(spec=[])  # no chat_template attribute

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer_no_template
    ):
        model = VllmModel(model_name_or_path="test-model")

        with pytest.raises(NotImplementedError, match="chat models"):
            model.load_tokenizer()


def test_format_inputs(model_instance):
    """Test format_inputs properly formats messages."""
    batch = TextBatch(
        text=[[UserMessage(content="Hello, world!")]],
        target=None,
        metadata={},
    )

    formatted = model_instance.format_inputs(batch)

    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert "prompt" in formatted[0]


def test_format_inputs_batch(model_instance):
    """Test format_inputs handles batch of multiple samples."""
    batch = TextBatch(
        text=[
            [UserMessage(content="First message")],
            [UserMessage(content="Second message")],
        ],
        target=None,
        metadata={},
    )

    formatted = model_instance.format_inputs(batch)

    assert len(formatted) == 2
    assert all("prompt" in f for f in formatted)


def test_model_forward(mock_llm, mock_tokenizer):
    """Test model_forward generates text correctly."""
    with (
        patch.object(VllmModel, "load_model", return_value=mock_llm),
        patch.object(VllmModel, "load_tokenizer", return_value=mock_tokenizer),
    ):
        model = VllmModel(model_name_or_path="test-model")
        model.configure_model()

        batch = [{"prompt": "test prompt"}]
        result = model.model_forward(batch)

        assert "generated_text" in result
        assert result["generated_text"] == ["Generated response"]
        mock_llm.generate.assert_called_once()


def test_generate(mock_llm, mock_tokenizer):
    """Test full generation pipeline."""
    with (
        patch.object(VllmModel, "load_model", return_value=mock_llm),
        patch.object(VllmModel, "load_tokenizer", return_value=mock_tokenizer),
    ):
        model = VllmModel(
            model_name_or_path="test-model",
            system_prompt="You are a helpful assistant.",
        )
        model.configure_model()

        batch = TextBatch(
            text=[[UserMessage(content="Hello!")]],
            target=None,
            metadata={},
        )

        result = model(batch)

        assert result["generated_text"] == ["Generated response"]


def test_system_prompt_stored():
    """Test that system prompt is properly stored."""
    model = VllmModel(
        model_name_or_path="test-model",
        system_prompt="You are a helpful assistant.",
    )

    assert model.system_message is not None
    assert model.system_message.content == "You are a helpful assistant."


def test_system_prompt_none():
    """Test that no system prompt results in None system_message."""
    model = VllmModel(model_name_or_path="test-model")

    assert model.system_message is None


def test_chat_template_stored():
    """Test that chat_template parameter is stored correctly."""
    custom_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    model = VllmModel(
        model_name_or_path="test-model",
        chat_template=custom_template,
    )

    assert model.chat_template == custom_template


def test_chat_template_none_by_default():
    """Test that chat_template is None by default."""
    model = VllmModel(model_name_or_path="test-model")

    assert model.chat_template is None


def test_chat_template_applied_to_tokenizer():
    """Test that custom chat_template is applied to the tokenizer."""
    custom_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None  # Initially no template

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        model = VllmModel(
            model_name_or_path="test-model",
            chat_template=custom_template,
        )
        tokenizer = model.load_tokenizer()

        assert tokenizer.chat_template == custom_template  # type: ignore[attr-defined]
