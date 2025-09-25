"""Models from Alibaba."""

import torch

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
                "max_new_tokens": 512,
                "do_sample": False,
            },
            processor_kwargs={
                "padding": True,
                "padding_side": "left",
                "max_pixels": 451584,  # 672*672
            },
            system_prompt=system_prompt,
            image_key="images",
        )
