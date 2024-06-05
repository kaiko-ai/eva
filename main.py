from eva.vision.data.datasets.segmentation.total_segmentator import TotalSegmentator2D

dataset = TotalSegmentator2D(
    root="data/total_segmentator",
    split="train",
    download=False,
)
dataset.setup()


for i in range(1000):
    filename = dataset.filename(i)
    print(filename)

