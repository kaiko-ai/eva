from eva.vision.data import datasets

import numpy as np

from eva.vision.data.transforms.normalization import functional


import torch


data = datasets.TotalSegmentator2D("data/total_segmentator", "train")
data.setup()

image_tensor, mask = data[0]

# image_tensor_clipped = image_tensor.clamp(min=-1024., max=1024.)
# actual = functional.rescale_intensity(image_tensor_clipped)
# print(actual.min().item(), actual.mean().item(), actual.max().item())


# from eva.vision.data import transforms

# first = transforms.Clamp((-1024, 1024))
# second = transforms.RescaleIntensity(out_range=(0., 1.))

# out = first(image_tensor)
# actual = second(out)
# print(actual.min().item(), actual.mean().item(), actual.max().item())


from eva.vision.data import transforms
from torchvision import tv_tensors

ts = transforms.CTScanTransforms()
im_ts = tv_tensors.Image(image_tensor)
tv_tensors.Image._to_tensor(im_ts)
mask_ts = tv_tensors.Mask(mask)
actual, actual_mask = ts(im_ts, mask_ts)
print(actual.min().item(), actual.mean().item(), actual.max().item())
print(actual_mask.min().item(), actual_mask.max().item(), actual_mask.unique())

print(type(actual_mask), actual_mask.dtype)

# from torchvision.transforms import functional

# image_tensor_clipped = image_tensor.clamp(min=-1024, max=1024)
# actual = functional.convert_image_dtype(image_tensor_clipped)
# print(actual.min().item(), actual.mean().item(), actual.max().item())
