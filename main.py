from eva.vision.models.networks.adapters import ViTAdapter
from eva.vision.models import ModelFromRegistry
import timm.models.vision_transformer
import torch
import timm
from eva.vision.models import TimmModel


# vit_backbone = timm.models.vision_transformer.VisionTransformer()
# vit_backbone = ModelFromRegistry(model_name="pathology/kaiko_vits16")
vit_backbone = TimmModel(model_name="vit_small_patch16_224", pretrained=False)

# vit_names = ["vit_tiny_patch16_224", "vit_small_patch8_224", "vit_small_patch16_224", "vit_base_patch8_224", "vit_base_patch16_224"]
# vit_backbone = timm.create_model(model_name=vit_names[0], pretrained=False)

adapter = ViTAdapter(
    vit_backbone=vit_backbone,
    interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]
)

x = torch.randn(1, 3, 224, 224)
out = adapter(x)

print(len(out))
for i in range(len(out)):
    print(out[i].shape)