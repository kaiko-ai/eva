# PubMedQA Text Classification

This tutorial demonstrates how to evaluate large language models on the PubMedQA dataset using eva's language module. PubMedQA is a biomedical question-answering dataset where models must classify answers as "yes", "no", or "maybe" based on medical abstracts and questions.

### Before you start

If you haven't downloaded the config files yet, please download them from [GitHub](https://github.com/kaiko-ai/eva/tree/main).

For this tutorial we use the [PubMedQA](https://pubmedqa.github.io/) dataset, which contains biomedical questions paired with abstracts from PubMed. The task is to classify whether the abstract supports a "yes", "no", or "maybe" answer to the question.

**Important**: This evaluation uses the manually gathered test set of 1000 questions from PubMedQA, which provides high-quality expert annotations for reliable model evaluation.

To let *eva* automatically handle the dataset download, the config file is already set with `download: true` in `configs/language/pubmedqa.yaml`. Additionally, you can set `DATA_ROOT` to configure the location where the dataset will be downloaded to / loaded from during evaluation (the default is `./data/pubmedqa`).

## Model Options

Eva supports three different ways to run language models:

1. **LiteLLM**: For API-based models (OpenAI, Anthropic, Together.ai, etc.) - requires API keys
2. **HuggingFace**: For running models directly on your computer (typically models under 8B parameters)
3. **vLLM**: For larger models that require cloud infrastructure (user must set up the environment)

## Running PubMedQA Classification

### 1. Using LiteLLM (API-based models)

First, set up your API key for the provider you want to use:

```bash
# For OpenAI models
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic models  
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For Together.ai models
export TOGETHER_API_KEY=your_together_api_key
```

Then run with provider-prefixed model names:

```bash
# Anthropic Claude models
MODEL_NAME=anthropic/claude-3-sonnet-20240229 eva fit --config configs/language/pubmedqa.yaml
MODEL_NAME=anthropic/claude-3-haiku-20240307 eva fit --config configs/language/pubmedqa.yaml

# OpenAI models
MODEL_NAME=openai/gpt-4o-mini eva fit --config configs/language/pubmedqa.yaml
MODEL_NAME=openai/gpt-4o eva fit --config configs/language/pubmedqa.yaml

# Together.ai models
MODEL_NAME=together_ai/meta-llama/Llama-2-7b-chat-hf eva fit --config configs/language/pubmedqa.yaml
```

### 2. Using HuggingFace models (local execution)

For smaller models (typically under 8B parameters) that can run locally on your machine:

First, update the config to use the HuggingFace wrapper:

```yaml
model:
  class_path: eva.language.models.TextModule
  init_args:
    model:
      class_path: eva.language.models.HuggingFaceTextModel
      init_args:
        model_name_or_path: microsoft/DialoGPT-medium
```

Then run:

```bash
MODEL_NAME=microsoft/DialoGPT-medium eva fit --config configs/language/pubmedqa.yaml
```

### 3. Using vLLM (cloud/distributed execution)

For larger models that require specialized infrastructure, you'll need to:

1. Set up a vLLM server in your cloud environment
2. Update the config to use the vLLM wrapper:

```yaml
model:
  class_path: eva.language.models.TextModule
  init_args:
    model:
      class_path: eva.language.models.VLLMTextModel
      init_args:
        model_name_or_path: meta-llama/Llama-2-70b-chat-hf
        server_url: http://your-vllm-server:8000
```

### 4. Basic evaluation with default configuration

The default PubMedQA config uses LiteLLM with Anthropic's Claude model. Run:

```bash
eva fit --config configs/language/pubmedqa.yaml
```

This command will:

- Download the PubMedQA dataset to `./data/pubmedqa` (if not already downloaded)
- Load the manually curated test set of 1000 question-abstract pairs
- Use the default Claude model to classify each question-abstract pair
- Store evaluation results including accuracy, precision, recall, and F1 scores

### 5. Customizing batch size and workers

For better performance or to work within API rate limits, you can adjust the batch size and number of workers:

```bash
BATCH_SIZE=4 N_DATA_WORKERS=2 eva fit --config configs/language/pubmedqa.yaml
```

## Understanding the results

Once the evaluation is complete:

- Check the evaluation results in `logs/<model-name>/pubmedqa/<session-id>/results.json`
- The results will include metrics computed on the 1000 manually annotated test examples:
  - **Accuracy**: Overall classification accuracy across all three classes
  - **Precision/Recall/F1**: Per-class and macro-averaged metrics
  - **Confusion Matrix**: Detailed breakdown of predictions vs. ground truth

## Key configuration components

The PubMedQA config demonstrates several important concepts:

#### Text prompting:
```yaml
prompt: "Instruction:\n Respond to the question with a single digit only: 0 for no, 1 for yes, or 2 for maybe. Do not include any words, explanations, or additional characters—only the digit."
```

#### Model configuration (LiteLLM):
```yaml
model:
  class_path: eva.language.models.LiteLLMTextModel
  init_args:
    model_name_or_path: ${oc.env:MODEL_NAME, anthropic/claude-3-7-sonnet-latest}
    model_kwargs:
      temperature: 0.01
```

#### Postprocessing:
```yaml
postprocess:
  predictions_transforms:
    - class_path: eva.language.utils.str_to_int_tensor.CastStrToIntTensor
```

This converts the model's text output to integer tensors for evaluation.

## Advanced usage

### Custom prompts

You can experiment with different prompting strategies by modifying the prompt in the config file. For example, you might try:

- Chain-of-thought prompting
- Few-shot examples
- Different output formats

### Model comparison

Run evaluations with multiple models to compare their performance on the 1000-question test set:

```bash
# Compare different API providers
MODEL_NAME=anthropic/claude-3-sonnet-20240229 eva fit --config configs/language/pubmedqa.yaml
MODEL_NAME=openai/gpt-4o eva fit --config configs/language/pubmedqa.yaml

# Compare model sizes within a provider
MODEL_NAME=anthropic/claude-3-haiku-20240307 eva fit --config configs/language/pubmedqa.yaml
MODEL_NAME=anthropic/claude-3-sonnet-20240229 eva fit --config configs/language/pubmedqa.yaml
```

The results from each run will be stored separately, allowing you to compare performance across different models and configurations.

## Notes

- **Test Set**: The evaluation uses the manually gathered 1000 test questions, ensuring high-quality annotations for reliable benchmarking
- **API Keys**: Make sure you have the appropriate API keys set up for the models you want to use and set them as environment variables
- **Model Names**: For LiteLLM, use provider-prefixed names (e.g., `anthropic/claude-3-sonnet-20240229`, `openai/gpt-4o`)
- **Rate Limits**: Be mindful of API rate limits when using commercial language models
- **Cost**: Commercial API usage incurs costs - consider using smaller batch sizes or local models for experimentation
- **Hardware Requirements**: Local HuggingFace models require sufficient GPU memory; vLLM setup requires cloud infrastructure management