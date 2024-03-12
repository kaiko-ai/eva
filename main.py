from eva.vision.data import datasets
from torchvision import transforms

data = datasets.TotalSegmentator2D(
    root="data/total_segmentator",
    split="train",
    # image_transforms=transforms.ToTensor()
)
data.prepare_data()
data.setup()
for i in range(1, 100, 10):
    image, mask = data[i]
    print(image, mask)
    quit()
