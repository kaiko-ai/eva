from eva.vision.models.networks.decoders.segmentation import DenselyDecoder
from eva.vision.models.networks.decoders.segmentation import densely

import torch

x = torch.rand(64, 384, 14, 14)


nn = densely.UpsampleDouble(384, 5)
out = nn(x)
print(out.shape)


nn = densely.CALayer(384, reduction=8)
out = nn(x)
print(out.shape)


nn = densely.DenseNetBlock(384, growth_rate=16)
out = nn(x)
print(out.shape)


nn = densely.DenseNetLayer(384, growth_rate=16, steps=1)
out = nn(x)
print(out.shape)





# decoder = DenselyDecoder(384, 5)
# decoder([x], (224, 224))
