import torch
from torch import nn

from eva.vision.models.networks import backbones, decoders

# backbone = backbones.TimmBackbone(
#     model_name="vit_small_patch16_224",
#     pretrained=True,
#     dynamic_img_size=True,    
# )
# embedding_size = 384
# num_classes = 5
# grid_size = (14, 14)

# linear_decoder = decoders.LinearDecoder(
#     network=nn.Linear(embedding_size, num_classes),
#     grid_size=grid_size,
# )
# conv_decoder = decoders.ConvDecoder(
#     network=nn.Conv2d(embedding_size, num_classes, kernel_size=(1, 1)),
#     grid_size=grid_size,
# )
# conv_decoder = decoders.ConvDecoder(
#     network=nn.Sequential(
#         nn.Upsample(scale_factor=2),
#         nn.Conv2d(embedding_size, 64, kernel_size=(3,3), padding=(1,1)),
#         nn.Upsample(scale_factor=2),
#         nn.Conv2d(64, num_classes, kernel_size=(3,3), padding=(1,1)),
#     ),
#     grid_size=grid_size,
# )

# def forward(image_tensor, backbone, decoder):
#     _, _, *image_size = image_tensor.shape
#     patch_embeddings = backbone(image_tensor)
#     logits = decoder(patch_embeddings, image_size)
#     return logits

# image_tensor = torch.Tensor(2, 3, 224, 224)
# linear_logits = forward(image_tensor, backbone, linear_decoder)
# conv_logits = forward(image_tensor, backbone, conv_decoder)

# print(linear_logits.shape)
# print(conv_logits.shape)
# quit()


import torchvision
from eva.vision.data import datasets
from eva.vision.data.transforms import common

from torchvision.transforms import v2


transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=224),
    v2.CenterCrop(size=224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

# target = tv_tensors.Mask(target)

# dataset = datasets.TotalSegmentator2D(
#     root="data/total_segmentator", split="train", download=False, image_target_transforms=transforms,
# )
dataset = datasets.TotalSegmentator2D(
    root="data/total_segmentator",
    split="train",
    download=False,
    transforms=transforms,
)

dataset.prepare_data()
dataset.setup()


image, masks = dataset[0]
print(type(image), image.shape, image.min(), image.max())
print(type(masks), masks.shape, masks.min(), masks.max())
