"""Models from Alibaba."""

import os

import torch

from eva.language.models.constants import MAX_NEW_TOKENS
from eva.language.utils import imports as import_utils
from eva.multimodal.models import wrappers
from eva.multimodal.models.networks.registry import model_registry


@model_registry.register("alibaba/qwen2-5-vl-7b-instruct")
class Qwen25VL7BInstruct(wrappers.HuggingFaceModel):
    """Qwen2.5-VL 7B Instruct model."""

    def __init__(
        self,
        system_prompt: str | None = None,
        cache_dir: str | None = None,
        attn_implementation: str = "flash_attention_2",
    ):
        """Initialize the model."""
        super().__init__(
            model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            model_class="Qwen2_5_VLForConditionalGeneration",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "cache_dir": cache_dir,
                "attn_implementation": attn_implementation,
            },
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
            },
            processor_kwargs={
                "padding": True,
                "padding_side": "left",
                "max_pixels": 451584,  # 672*672
            },
            system_prompt=system_prompt,
            image_key="images",
            image_position="before_text",
        )


if import_utils.is_vllm_available():

    @model_registry.register("alibaba/qwen2-5-vl-7b-instruct-vllm")
    class Qwen25VL7BInstructVllm(wrappers.VllmModel):
        """Qwen2.5-VL 7B Instruct model."""

        def __init__(
            self,
            system_prompt: str | None = None,
        ):
            """Initialize the model."""
            super().__init__(
                model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
                model_kwargs={
                    "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", 1)),
                    "max_num_seqs": 8,
                    "mm_processor_kwargs": {"use_fast": False},
                    "enable_prefix_caching": False,
                },
                image_position="before_text",
                system_prompt=system_prompt,
            )

    @model_registry.register("alibaba/qwen2-5-vl-72b-instruct-vllm")
    class Qwen25VL72BInstructVLLM(wrappers.VllmModel):
        """Qwen2.5-VL 72B Instruct model."""

        def __init__(
            self,
            system_prompt: str | None = None,
        ):
            """Initialize the model."""
            super().__init__(
                model_name_or_path="Qwen/Qwen2.5-VL-72B-Instruct",
                model_kwargs={
                    "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", 4)),
                    "max_num_seqs": 8,
                    "mm_processor_kwargs": {"use_fast": False},
                    "enable_prefix_caching": False,
                },
                image_position="before_text",
                system_prompt=system_prompt,
            )
