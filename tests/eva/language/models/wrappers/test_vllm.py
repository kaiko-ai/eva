"""VLLM wrapper tests."""

import pytest

try:
    from eva.language.models.wrappers.vllm import VLLMTextModel
except ImportError:
    pytest.skip("vLLM not available", allow_module_level=True)


class MockLLM:
    """Mock vLLM LLM class."""

    def __init__(self, model: str, **kwargs):
        """Initialize mock LLM with model name and kwargs."""
        self.model = model
        self.kwargs = kwargs
        self._tokenizer = MockTokenizer()

    def get_tokenizer(self):
        """Return the mock tokenizer."""
        return self._tokenizer

    def generate(self, prompts, sampling_params):
        """Generate mock responses for given prompts."""
        return [MockOutput() for _ in prompts]


class MockTokenizer:
    """Mock vLLM tokenizer."""

    def __init__(self):
        """Initialize mock tokenizer with default properties."""
        self.chat_template = "template"
        self.bos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        """Apply mock chat template returning dummy token lists."""
        # Return list of token lists for each message
        return [[1, 2, 3, 4] for _ in messages]


class MockTokensPrompt:
    """Mock vLLM TokensPrompt."""

    def __init__(self, prompt_token_ids):
        """Initialize with token IDs."""
        self.prompt_token_ids = prompt_token_ids


class MockOutput:
    """Mock vLLM output."""

    def __init__(self):
        """Initialize with mock text outputs."""
        self.outputs = [MockTextOutput()]


class MockTextOutput:
    """Mock vLLM text output."""

    def __init__(self):
        """Initialize with default response text."""
        self.text = "Generated response"


class MockSamplingParams:
    """Mock vLLM SamplingParams."""

    def __init__(self, **kwargs):
        """Initialize with sampling parameters."""
        self.kwargs = kwargs


@pytest.fixture
def mock_vllm_imports(monkeypatch):
    """Mock vLLM imports to avoid dependency."""
    monkeypatch.setattr("eva.language.models.wrappers.vllm.LLM", MockLLM)
    monkeypatch.setattr("eva.language.models.wrappers.vllm.SamplingParams", MockSamplingParams)
    monkeypatch.setattr("eva.language.models.wrappers.vllm.TokensPrompt", MockTokensPrompt)


def test_initialization(mock_vllm_imports):
    """Tests VLLMTextModel initialization."""
    model = VLLMTextModel(
        model_name_or_path="test/model",
        model_kwargs={"max_model_len": 1024},
        generation_kwargs={"max_tokens": 100},
    )
    assert model._model_name_or_path == "test/model"
    assert model._model_kwargs == {"max_model_len": 1024}
    assert model._generation_kwargs == {"max_tokens": 100}
    assert model._llm_model is None


def test_lazy_loading(mock_vllm_imports):
    """Tests lazy model loading."""
    model = VLLMTextModel("test/model")
    assert model._llm_model is None

    model.load_model()
    assert model._llm_model is not None
    assert model._llm_tokenizer is not None


def test_generate(mock_vllm_imports):
    """Tests text generation."""
    model = VLLMTextModel("test/model")
    prompts = ["Hello", "How are you?"]

    results = model(prompts)

    assert len(results) == 2
    assert all(result == "Generated response" for result in results)


def test_chat_template_application(mock_vllm_imports):
    """Tests chat template application."""
    model = VLLMTextModel("test/model")
    prompts = ["Hello world"]

    token_prompts = model._apply_chat_template(prompts)

    assert len(token_prompts) == 1
    assert hasattr(token_prompts[0], "prompt_token_ids")


def test_double_bos_removal(mock_vllm_imports, monkeypatch):
    """Tests double BOS token removal warning is triggered."""

    class MockTokenizerDoubleBOS(MockTokenizer):
        def apply_chat_template(
            self, _messages, tokenize=True, add_generation_prompt=True
        ):  # noqa: ARG002
            return [1, 1, 2, 3, 4]

    def mock_get_tokenizer():
        return MockTokenizerDoubleBOS()

    model = VLLMTextModel("test/model")
    model.load_model()
    monkeypatch.setattr(model._llm_model, "get_tokenizer", mock_get_tokenizer)
    model._llm_tokenizer = mock_get_tokenizer()

    with monkeypatch.context() as m:
        # Mock logger to capture warning
        log_messages = []

        def mock_warning(msg):
            log_messages.append(msg)

        m.setattr("eva.language.models.wrappers.vllm.logger.warning", mock_warning)

        token_prompts = model._apply_chat_template(["test"])

        assert any("double start token" in msg for msg in log_messages)
        assert len(token_prompts) > 0


def test_no_chat_template_error(mock_vllm_imports, monkeypatch):
    """Tests error when tokenizer has no chat template."""

    class MockTokenizerNoTemplate:
        def __init__(self):
            """Initialize tokenizer without chat template."""
            pass

    model = VLLMTextModel("test/model")
    model.load_model()
    monkeypatch.setattr(model._llm_model, "get_tokenizer", lambda: MockTokenizerNoTemplate())
    model._llm_tokenizer = MockTokenizerNoTemplate()

    with pytest.raises(ValueError, match="does not have a chat template"):
        model._apply_chat_template(["test"])
