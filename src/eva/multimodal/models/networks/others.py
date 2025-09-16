"""Models from other providers (non-major entities)."""

import os

import torch

from eva.core.utils import requirements
from eva.multimodal.models import wrappers
from eva.multimodal.models.networks.registry import model_registry


@model_registry.register("others/wenchuanzhang_patho-r1-3b")
class PathoR13b(wrappers.HuggingFaceModel):
    """Patho-R1-3B model by Wenchuan Zhang."""

    def __init__(
        self,
        system_prompt: str | None = None,
        cache_dir: str | None = None,
        attn_implementation: str = "flash_attention_2",
    ):
        """Initialize the Patho-R1-3B model."""
        requirements.check_min_versions(requirements={"torch": "2.5.1", "torchvision": "0.20.1"})

        if not os.getenv("HF_TOKEN"):
            raise ValueError("HF_TOKEN env variable must be set.")

        super().__init__(
            model_name_or_path="WenchuanZhang/Patho-R1-3B",
            model_class="Qwen2_5_VLForConditionalGeneration",
            model_kwargs={
                "torch_dtype": torch.float16,
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
