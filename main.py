from eva.vision.data.datasets import segmentation

# dataset = segmentation.TotalSegmentator2D(
#     root="data/total_segmentator",
#     split="train",
#     download=False,
# )
# dataset.setup()

dataset = segmentation.EmbeddingsSegmentationDataset(
    root="./data/embeddings/dino_vits16/total_segmentator_2d",
    split="train",
    manifest_file="manifest.csv",
)
dataset.setup()

for i in range(100):
    data, targets = dataset[i]
    print(data)
    print(targets)
    quit()
