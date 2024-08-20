import torch
import timm

model = timm.create_model(
    model_name="resnet50",
    features_only=True,
    out_indices=[0, 1, 3]
)

model = timm.create_model(
    model_name="vit_small_patch16_224",
    features_only=True,
    out_indices=[0, 1, 3]
)

x = torch.rand(2, 3, 224, 224)
out = model(x)
print(out[0].shape)