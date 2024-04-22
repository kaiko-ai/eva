import transformers
import torch
from eva.vision.models.networks import huggingface

model = transformers.Mask2FormerForUniversalSegmentation(
    config=transformers.Mask2FormerConfig(
        backbone_config=huggingface.TimmBackboneConfigV2(
            backbone="vit_small_patch16_224",
            output_hidden_states=True,
            out_indices=(8, 9, 10, 11),
            backbone_kwargs={
                "dynamic_img_size": True,
            }
        ),
    ),
)

tensor = torch.Tensor(1, 3, 224, 224)
out = model(pixel_values=tensor)
print(out)

