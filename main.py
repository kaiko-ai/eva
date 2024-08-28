import torch
import timm

# backbone = timm.create_model(
#     model_name="resnet18",
#     features_only=True,
#     # out_indices=(0, 1, 2, 3, 4),
#     out_indices=(0, 2, 4),
#     # out_indices=(2, 3, 4),
# )

# x = torch.rand(2, 3, 224, 224)
# out = backbone(x)
# print(out[0].shape, out[1].shape, out[2].shape)


backbone = timm.create_model(
    model_name="swin_tiny_patch4_window7_224",
    features_only=True,
    # out_indices=(0, 1, 2, 3, 4),
    # out_indices=(0, 1, 2),
    out_indices=(0, 2, 3),
)

x = torch.rand(2, 3, 224, 224)
out = backbone(x)
print(out[0].shape, out[1].shape, out[2].shape)
