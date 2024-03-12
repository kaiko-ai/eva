from eva.vision.data import datasets
from torchvision import transforms
import numpy as np

# segmentation = datasets.TotalSegmentator2D(
#     root="data/total_segmentator",
#     split="train",
#     # image_transforms=transforms.ToTensor()
# )
# segmentation.prepare_data()
# segmentation.setup()
# for i in range(1, 100, 10):
#     image, mask = segmentation[i]
#     quit()


classification = datasets.TotalSegmentatorClassification(
    root="data/total_segmentator",
    split="train",
    # image_transforms=transforms.ToTensor()
)
classification.prepare_data()
classification.setup()
for i in range(1, 100, 10):
    image, mask = classification[i]
    print(mask)
    quit()
